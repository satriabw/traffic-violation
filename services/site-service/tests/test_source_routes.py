import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from shared.models.source import SourceMetadata
from shared.video.probe import ProbeUnavailable, VideoUnreadable

from site_service.main import app, get_db
from site_service.storage import get_storage
from site_service.video import get_probe
from tests.test_file_service import FakeStorage

SITES = "/api/v1/sites"

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=64)


class FakeProbe:
    """Stands in for shared.video.probe.probe. Records the URLs it was asked about so
    a test can assert a stream was never probed, which would mean a live RTSP connect
    inside a request."""

    def __init__(self):
        self.calls: list[str] = []
        self.error: Exception | None = None
        self.result = SourceMetadata(
            total_frames=27000, fps=29.97, nominal_fps=30.0,
            duration_seconds=900.0, resolution={"width": 1280, "height": 720},
        )

    def __call__(self, url: str) -> SourceMetadata:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.result


_probe = FakeProbe()


def override_get_db():
    yield _test_con


def override_get_storage():
    yield _storage


def override_get_probe():
    yield _probe


client = TestClient(app)


@pytest.fixture(autouse=True)
def _dependency_overrides():
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage] = override_get_storage
    app.dependency_overrides[get_probe] = override_get_probe
    _probe.calls.clear()
    _probe.error = None
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def site_id():
    return client.post(SITES, json={"name": "Junction 5"}).json()["id"]


def _sources(site_id: str) -> str:
    return f"{SITES}/{site_id}/sources"


def _stream(url="rtsp://10.0.0.5/s"):
    return {"kind": "stream", "stream_url": url}


def _video_file(*, uploaded: bool = True, file_type: str = "video") -> str:
    created = client.post(
        "/api/v1/files", json={"name": "a.mp4", "type": file_type, "size_bytes": 64}
    ).json()
    if uploaded:
        client.post(f"/api/v1/files/{created['id']}/complete")
    return created["id"]


def test_unprefixed_sources_path_is_not_served(site_id):
    assert client.get(f"/sites/{site_id}/sources").status_code == 404


def test_create_stream_source_returns_201(site_id):
    response = client.post(_sources(site_id), json=_stream())

    assert response.status_code == 201
    body = response.json()
    assert body["version"] == 1
    assert body["kind"] == "stream"
    assert body["status"] == "created"
    assert body["file_id"] is None


def test_create_video_source_returns_201(site_id):
    file_id = _video_file()

    response = client.post(_sources(site_id), json={"kind": "video", "file_id": file_id})

    assert response.status_code == 201
    body = response.json()
    assert body["file_id"] == file_id
    assert body["stream_url"] is None


def test_changing_the_source_appends_a_version(site_id):
    client.post(_sources(site_id), json=_stream("rtsp://a"))
    second = client.post(_sources(site_id), json=_stream("rtsp://b"))

    assert second.json()["version"] == 2


def test_a_site_may_hold_both_kinds_over_time(site_id):
    # A camera you stream and also upload recordings from is one location.
    client.post(_sources(site_id), json=_stream())
    client.post(_sources(site_id), json={"kind": "video", "file_id": _video_file()})

    assert client.get(_sources(site_id)).json()["kind"] == "video"


def test_get_returns_the_active_source(site_id):
    client.post(_sources(site_id), json=_stream("rtsp://a"))
    latest = client.post(_sources(site_id), json=_stream("rtsp://b")).json()

    assert client.get(_sources(site_id)).json()["id"] == latest["id"]


def test_superseded_versions_stay_addressable(site_id):
    first = client.post(_sources(site_id), json=_stream("rtsp://a")).json()
    client.post(_sources(site_id), json=_stream("rtsp://b"))

    response = client.get(f"{_sources(site_id)}/{first['id']}")

    assert response.status_code == 200
    assert response.json()["stream_url"] == "rtsp://a"


def test_get_returns_404_when_the_site_has_no_source(site_id):
    response = client.get(_sources(site_id))

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found"}


def test_get_by_id_is_scoped_to_its_site(site_id):
    created = client.post(_sources(site_id), json=_stream()).json()
    other = client.post(SITES, json={"name": "Other"}).json()["id"]

    assert client.get(f"{_sources(site_id)}/{created['id']}").status_code == 200
    assert client.get(f"{_sources(other)}/{created['id']}").status_code == 404


def test_create_returns_404_for_an_unknown_site():
    response = client.post(_sources("no-such-site"), json=_stream())

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_video_source_returns_409_when_the_file_was_never_uploaded(site_id):
    # The site would otherwise be pointed at a reserved slot with no bytes behind it.
    file_id = _video_file(uploaded=False)

    response = client.post(_sources(site_id), json={"kind": "video", "file_id": file_id})

    assert response.status_code == 409


def test_video_source_returns_422_for_an_unknown_file(site_id):
    response = client.post(
        _sources(site_id), json={"kind": "video", "file_id": "no-such-file"}
    )

    assert response.status_code == 422


def test_video_source_returns_422_for_a_file_of_the_wrong_type(site_id):
    response = client.post(
        _sources(site_id),
        json={"kind": "video", "file_id": _video_file(file_type="calibration")},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "label,body",
    [
        ("video with an address", {"kind": "video", "stream_url": "rtsp://x"}),
        ("video with nothing", {"kind": "video"}),
        ("stream with a file", {"kind": "stream", "file_id": "f"}),
        ("stream with nothing", {"kind": "stream"}),
        ("unknown kind", {"kind": "carrier-pigeon", "stream_url": "rtsp://x"}),
        ("site_id in the body", {"kind": "stream", "stream_url": "rtsp://x", "site_id": "x"}),
    ],
)
def test_create_returns_422_for_a_malformed_body(site_id, label, body):
    assert client.post(_sources(site_id), json=body).status_code == 422


def test_a_stream_source_needs_no_file_at_all(site_id):
    # Its address is a URL this service never resolves, so there is nothing to verify.
    assert client.post(_sources(site_id), json=_stream()).status_code == 201


def test_creating_a_video_source_stores_the_probed_metadata(site_id):
    # Read once at creation so nothing downstream has to seek the file again.
    response = client.post(
        _sources(site_id), json={"kind": "video", "file_id": _video_file()}
    )

    meta = response.json()["metadata"]
    assert meta["fps"] == 29.97
    assert meta["nominal_fps"] == 30.0
    assert meta["total_frames"] == 27000
    assert meta["resolution"] == {"width": 1280, "height": 720}


def test_creating_a_stream_source_never_probes(site_id):
    # A live feed has no index to read, and connecting to one here would block the
    # request on a camera that may not even be up yet.
    response = client.post(_sources(site_id), json=_stream())

    assert response.status_code == 201
    assert response.json()["metadata"] is None
    assert _probe.calls == []


def test_an_undecodable_video_is_rejected(site_id):
    _probe.error = VideoUnreadable("moov atom not found")

    response = client.post(
        _sources(site_id), json={"kind": "video", "file_id": _video_file()}
    )

    assert response.status_code == 422
    # Rejected outright rather than stored: a source nothing can decode would only
    # fail again once detection jobs were enqueued against it.
    assert client.get(_sources(site_id)).status_code == 404


def test_a_probe_that_could_not_reach_storage_is_not_the_clients_fault(site_id):
    _probe.error = ProbeUnavailable("Connection refused")

    response = client.post(
        _sources(site_id), json={"kind": "video", "file_id": _video_file()}
    )

    assert response.status_code == 502
    assert client.get(_sources(site_id)).status_code == 404


def test_a_video_source_given_inline_on_the_site_is_probed_too():
    # POST /sites with an inline source and POST /sites/{id}/sources must not drift:
    # the inline form is sugar, not a second path that skips the probe.
    file_id = _video_file()

    response = client.post(
        SITES, json={"name": "Junction 6", "source": {"kind": "video", "file_id": file_id}}
    )

    assert response.status_code == 201
    assert response.json()["source"]["metadata"]["fps"] == 29.97
