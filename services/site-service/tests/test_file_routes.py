import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from shared.config import S3_MAX_UPLOAD_BYTES as MAX_UPLOAD_BYTES

from site_service.main import app, get_db
from site_service.storage import get_storage
from tests.test_file_service import FakeStorage

# Spelled out rather than imported from shared.config so the tests pin the wire path
# clients actually call.
FILES = "/api/v1/files"

_test_con = get_connection(":memory:")
init_db(_test_con)

# Mutable so a test can decide whether the object "exists" in storage.
_storage = FakeStorage()


def override_get_db():
    yield _test_con


def override_get_storage():
    yield _storage


client = TestClient(app)

@pytest.fixture(autouse=True)
def _dependency_overrides():
    # app.dependency_overrides is global to the shared app object, so each module has
    # to install its own and tear them down — otherwise import order decides which
    # module's connection and storage every test gets.
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage] = override_get_storage
    yield
    app.dependency_overrides.clear()



@pytest.fixture(autouse=True)
def reset_storage():
    _storage.existing_size = None
    yield


def _create(**overrides):
    return client.post(
        FILES, json={"name": "clip.mp4", "type": "video", "size_bytes": 4096, **overrides}
    )


def test_unprefixed_files_path_is_not_served():
    assert client.post(
        "/files", json={"name": "a", "type": "video", "size_bytes": 1}
    ).status_code == 404


def test_create_file_returns_201_with_a_pending_row():
    response = _create()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["url"] == f"video/{body['id']}/clip.mp4"


def test_create_file_returns_an_upload_url():
    body = _create(content_type="video/mp4").json()

    assert body["upload_url"].startswith("https://r2.test/put/")
    assert body["upload_expires_in"] > 0


def test_create_file_returns_422_on_an_unknown_type():
    assert _create(type="spreadsheet").status_code == 422


def test_create_file_returns_422_without_a_name():
    assert client.post(FILES, json={"type": "video", "size_bytes": 1}).status_code == 422


def test_complete_returns_404_for_an_unknown_file():
    # An unknown id is not a storage problem, so it must not surface as 409.
    assert client.post(f"{FILES}/no-such-file/complete").status_code == 404


def test_complete_returns_409_when_the_object_never_landed():
    file_id = _create().json()["id"]

    response = client.post(f"{FILES}/{file_id}/complete")

    assert response.status_code == 409
    assert response.json()["detail"] == "Upload not found in storage"


def test_complete_returns_200_and_marks_the_file_uploaded():
    file_id = _create().json()["id"]
    _storage.existing_size = 4096

    body = client.post(f"{FILES}/{file_id}/complete").json()

    assert body["status"] == "uploaded"
    assert body["size_bytes"] == 4096


def test_get_file_returns_404_when_unknown():
    assert client.get(f"{FILES}/no-such-file").status_code == 404


def test_get_file_returns_200_with_a_download_url_once_uploaded():
    file_id = _create().json()["id"]
    _storage.existing_size = 4096
    client.post(f"{FILES}/{file_id}/complete")

    body = client.get(f"{FILES}/{file_id}").json()

    assert body["status"] == "uploaded"
    assert body["download_url"].startswith("https://r2.test/get/")


def test_get_file_has_no_download_url_while_pending():
    file_id = _create().json()["id"]

    assert client.get(f"{FILES}/{file_id}").json()["download_url"] is None


def test_get_file_never_returns_an_upload_url():
    file_id = _create().json()["id"]

    assert "upload_url" not in client.get(f"{FILES}/{file_id}").json()


def test_create_file_returns_413_when_over_the_size_cap():
    response = _create(size_bytes=MAX_UPLOAD_BYTES + 1)

    assert response.status_code == 413
    assert str(MAX_UPLOAD_BYTES) in response.json()["detail"]


def test_create_file_accepts_a_file_exactly_at_the_cap():
    assert _create(size_bytes=MAX_UPLOAD_BYTES).status_code == 201
