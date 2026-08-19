import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from site_service.main import app, get_db
from site_service.storage import get_storage
from site_service.video import get_probe
from tests.test_file_service import FakeStorage
from tests.test_source_routes import FakeProbe

# Spelled out rather than imported from shared.config so the tests pin the wire
# path clients actually call, and a change to the prefix has to be deliberate.
SITES = "/api/v1/sites"
STREAM = {"kind": "stream", "stream_url": "rtsp://10.0.0.5/s"}

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=64)
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
    # app.dependency_overrides is global to the shared app object, so each module has
    # to install its own and tear them down — otherwise import order decides which
    # module's connection and storage every test gets.
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage] = override_get_storage
    app.dependency_overrides[get_probe] = override_get_probe
    yield
    app.dependency_overrides.clear()


def _video_file(*, uploaded: bool = True) -> str:
    created = client.post(
        "/api/v1/files", json={"name": "a.mp4", "type": "video", "size_bytes": 64}
    ).json()
    if uploaded:
        client.post(f"/api/v1/files/{created['id']}/complete")
    return created["id"]


def test_health_is_served_outside_the_version_prefix():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unprefixed_resource_path_is_not_served():
    assert client.get("/sites").status_code == 404


def test_create_site_needs_only_a_name():
    response = client.post(SITES, json={"name": "Junction 5"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Junction 5"
    # A site can exist before anything is pointed at it.
    assert body["source"] is None


def test_create_site_returns_422_without_a_name():
    assert client.post(SITES, json={}).status_code == 422


def test_create_site_with_an_inline_stream_source():
    response = client.post(SITES, json={"name": "Junction 5", "source": STREAM})

    assert response.status_code == 201
    source = response.json()["source"]
    assert source["version"] == 1
    assert source["stream_url"] == "rtsp://10.0.0.5/s"


def test_create_site_with_an_inline_video_source():
    file_id = _video_file()

    response = client.post(
        SITES, json={"name": "Junction 5", "source": {"kind": "video", "file_id": file_id}}
    )

    assert response.status_code == 201
    assert response.json()["source"]["file_id"] == file_id


def test_inline_source_is_validated_like_the_dedicated_endpoint():
    # Same guard, same status — the two entry points must not drift apart.
    file_id = _video_file(uploaded=False)

    response = client.post(
        SITES, json={"name": "Junction 5", "source": {"kind": "video", "file_id": file_id}}
    )

    assert response.status_code == 409


def test_create_site_returns_422_for_an_invalid_inline_source():
    response = client.post(
        SITES,
        json={"name": "Junction 5", "source": {"kind": "video", "stream_url": "rtsp://x"}},
    )

    assert response.status_code == 422


def test_site_no_longer_accepts_a_top_level_source():
    # Sources moved to their own table; a site is a durable location.
    response = client.post(SITES, json={"name": "J", "mode": "stream", "url": "rtsp://x"})

    assert response.status_code == 422


def test_get_site_returns_200_when_found():
    created = client.post(SITES, json={"name": "Junction 5"}).json()

    response = client.get(f"{SITES}/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_site_returns_404_when_missing():
    response = client.get(f"{SITES}/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_get_site_embeds_the_active_source():
    site = client.post(SITES, json={"name": "J", "source": STREAM}).json()
    client.post(
        f"{SITES}/{site['id']}/sources",
        json={"kind": "stream", "stream_url": "rtsp://10.0.0.9/s"},
    )

    body = client.get(f"{SITES}/{site['id']}").json()

    assert body["source"]["version"] == 2
    assert body["source"]["stream_url"] == "rtsp://10.0.0.9/s"


def test_list_sites_returns_envelope():
    client.post(SITES, json={"name": "A"})
    client.post(SITES, json={"name": "B", "source": STREAM})

    response = client.get(SITES)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    assert "items" in body


def test_list_sites_filters_by_the_active_sources_kind():
    client.post(SITES, json={"name": "StreamSite", "source": STREAM})

    response = client.get(SITES, params={"kind": "stream"})

    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert all(item["source"]["kind"] == "stream" for item in items)


def test_list_sites_rejects_an_unknown_kind():
    assert client.get(SITES, params={"kind": "carrier-pigeon"}).status_code == 422


def test_delete_site_returns_204_when_found():
    created = client.post(SITES, json={"name": "ToDelete", "source": STREAM}).json()

    response = client.delete(f"{SITES}/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"{SITES}/{created['id']}").status_code == 404


def test_delete_site_returns_404_when_missing():
    response = client.delete(f"{SITES}/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def _create_site(name="CalSite"):
    return client.post(SITES, json={"name": name}).json()
