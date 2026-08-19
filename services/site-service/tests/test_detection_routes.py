import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.models.detection import ViolationType
from shared.models.source import SourceMetadata
from shared.queue.memory import InMemoryQueue

from site_service.main import app, get_db
from site_service.queue import get_queue
from site_service.storage import get_storage
from site_service.video import get_probe
from tests.test_file_service import FakeStorage
from tests.test_source_routes import FakeProbe

SITES = "/api/v1/sites"

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=64)
_probe = FakeProbe()
_queue = InMemoryQueue()

client = TestClient(app)


@pytest.fixture(autouse=True)
def _dependency_overrides():
    app.dependency_overrides[get_db] = lambda: _test_con
    app.dependency_overrides[get_storage] = lambda: _storage
    app.dependency_overrides[get_probe] = lambda: _probe
    # A real queue here would mean every test in this file needs Redis up.
    app.dependency_overrides[get_queue] = lambda: _queue
    _probe.result = SourceMetadata(total_frames=27000, fps=29.97)
    while _queue.consume() is not None:
        pass
    yield
    app.dependency_overrides.clear()


def _video_file() -> str:
    created = client.post(
        "/api/v1/files", json={"name": "a.mp4", "type": "video", "size_bytes": 64}
    ).json()
    client.post(f"/api/v1/files/{created['id']}/complete")
    return created["id"]


def _site_with_video() -> str:
    return client.post(
        SITES,
        json={"name": "Junction 5", "source": {"kind": "video", "file_id": _video_file()}},
    ).json()["id"]


def _site_with_stream() -> str:
    return client.post(
        SITES,
        json={"name": "Junction 5", "source": {"kind": "stream", "stream_url": "rtsp://a"}},
    ).json()["id"]


def _detect(site_id: str) -> str:
    return f"{SITES}/{site_id}/detect"


def test_unprefixed_detect_path_is_not_served():
    assert client.post(f"/sites/{_site_with_video()}/detect", json={}).status_code == 404


def test_detect_accepts_the_request_with_202():
    # 202, not 201: nothing was created, the work was accepted.
    response = client.post(_detect(_site_with_video()), json={})

    assert response.status_code == 202


def test_frame_range_comes_from_the_probed_metadata():
    body = client.post(_detect(_site_with_video()), json={}).json()

    # The whole video, in the frame indices the probe reported.
    assert body["frame_range"] == {"start": 0, "end": 27000}


def test_the_job_lands_on_the_queue():
    site_id = _site_with_video()

    body = client.post(_detect(site_id), json={}).json()

    queued = _queue.consume()
    assert queued is not None
    assert queued.id == body["id"]
    assert queued.site_id == site_id


def test_types_default_to_every_known_violation_type():
    body = client.post(_detect(_site_with_video()), json={}).json()

    assert body["types"] == [t.value for t in ViolationType]


def test_body_types_override_the_default():
    body = client.post(
        _detect(_site_with_video()), json={"types": ["red_light_running"]}
    ).json()

    assert body["types"] == ["red_light_running"]
    assert _queue.consume().types == [ViolationType.RED_LIGHT_RUNNING]


def test_an_empty_body_is_accepted():
    # types is the only field, and it is optional — POST with no body at all works.
    assert client.post(_detect(_site_with_video())).status_code == 202


def test_unknown_body_field_is_rejected():
    # frame_range is derived from the source; a client sending one should learn that.
    response = client.post(
        _detect(_site_with_video()), json={"frame_range": {"start": 0, "end": 10}}
    )

    assert response.status_code == 422


def test_unknown_violation_type_is_rejected():
    assert client.post(_detect(_site_with_video()), json={"types": ["jaywalking"]}).status_code == 422


def test_unknown_site_is_404():
    assert client.post(_detect("nope"), json={}).status_code == 404


def test_site_without_a_source_is_409():
    site_id = client.post(SITES, json={"name": "Junction 5"}).json()["id"]

    response = client.post(_detect(site_id), json={})

    assert response.status_code == 409
    assert response.json()["detail"] == "Site has no source"


def test_stream_source_is_409():
    # A live feed is the supervisor-worker path in the LLD, not this endpoint: there
    # is no frame count to bound a job with.
    response = client.post(_detect(_site_with_stream()), json={})

    assert response.status_code == 409
    assert "video source" in response.json()["detail"]


def test_video_without_a_frame_count_is_409():
    _probe.result = SourceMetadata(fps=29.97)  # probed, but nb_frames was absent

    response = client.post(_detect(_site_with_video()), json={})

    assert response.status_code == 409
    assert response.json()["detail"] == "Video metadata is not available"


def test_a_rejected_request_queues_nothing():
    client.post(_detect(_site_with_stream()), json={})

    assert _queue.consume() is None
