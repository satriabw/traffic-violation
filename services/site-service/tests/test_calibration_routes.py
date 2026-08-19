"""Route-level rules for both versioned-document resources.

Parameterised over calibrations and configurations: they are the same endpoint shape
over the same service functions, so a rule proven for one must hold for the other.
"""

import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from site_service.main import app, get_db
from site_service.storage import get_storage
from tests.test_file_service import FakeStorage

RESOURCES = [("calibrations", "calibration"), ("configurations", "configuration")]

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=128)


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



@pytest.fixture
def site_id():
    return client.post(
        "/api/v1/sites", json={"name": "Main St"}
    ).json()["id"]


def _file(file_type: str, *, uploaded: bool = True) -> str:
    created = client.post(
        "/api/v1/files", json={"name": "a.json", "type": file_type, "size_bytes": 128}
    ).json()
    if uploaded:
        client.post(f"/api/v1/files/{created['id']}/complete")
    return created["id"]


def _path(site_id: str, resource: str) -> str:
    return f"/api/v1/sites/{site_id}/{resource}"


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_201_with_version_one(site_id, resource, file_type):
    response = client.post(_path(site_id, resource), json={"file_id": _file(file_type)})

    assert response.status_code == 201
    assert response.json()["version"] == 1


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_each_post_appends_a_version(site_id, resource, file_type):
    client.post(_path(site_id, resource), json={"file_id": _file(file_type)})
    second = client.post(_path(site_id, resource), json={"file_id": _file(file_type)})

    assert second.json()["version"] == 2


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_get_returns_the_active_version(site_id, resource, file_type):
    client.post(_path(site_id, resource), json={"file_id": _file(file_type)})
    latest = client.post(_path(site_id, resource), json={"file_id": _file(file_type)}).json()

    assert client.get(_path(site_id, resource)).json()["id"] == latest["id"]


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_404_for_an_unknown_site(resource, file_type):
    response = client.post(_path("no-such-site", resource), json={"file_id": _file(file_type)})

    assert response.status_code == 404
    assert response.json()["detail"] == "Site not found"


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_422_for_an_unknown_file(site_id, resource, file_type):
    response = client.post(_path(site_id, resource), json={"file_id": "no-such-file"})

    assert response.status_code == 422


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_409_when_the_file_was_never_uploaded(site_id, resource, file_type):
    # A reserved slot is not a file. This is the guard the two-phase upload exists for.
    file_id = _file(file_type, uploaded=False)

    response = client.post(_path(site_id, resource), json={"file_id": file_id})

    assert response.status_code == 409
    assert "not been uploaded" in response.json()["detail"]


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_422_for_a_file_of_the_wrong_type(site_id, resource, file_type):
    response = client.post(_path(site_id, resource), json={"file_id": _file("video")})

    assert response.status_code == 422


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_create_returns_422_without_a_file_id(site_id, resource, file_type):
    assert client.post(_path(site_id, resource), json={}).status_code == 422


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_get_returns_404_when_the_site_has_none(site_id, resource, file_type):
    assert client.get(_path(site_id, resource)).status_code == 404


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_get_by_id_is_scoped_to_its_site(site_id, resource, file_type):
    created = client.post(_path(site_id, resource), json={"file_id": _file(file_type)}).json()
    other = client.post(
        "/api/v1/sites", json={"name": "Other"}
    ).json()["id"]

    assert client.get(f"{_path(site_id, resource)}/{created['id']}").status_code == 200
    assert client.get(f"{_path(other, resource)}/{created['id']}").status_code == 404


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_unprefixed_path_is_not_served(site_id, resource, file_type):
    # Pins the wire path: the version prefix is mounted by the service, not rewritten
    # by a gateway.
    assert client.get(f"/sites/{site_id}/{resource}").status_code == 404


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_get_active_returns_404_for_an_unknown_site(resource, file_type):
    response = client.get(_path("no-such-site", resource))

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_get_by_id_returns_a_superseded_version(site_id, resource, file_type):
    first = client.post(_path(site_id, resource), json={"file_id": _file(file_type)}).json()
    client.post(_path(site_id, resource), json={"file_id": _file(file_type)})

    # Superseded versions stay addressable — only the active one changes.
    response = client.get(f"{_path(site_id, resource)}/{first['id']}")

    assert response.status_code == 200
    assert response.json()["version"] == 1


@pytest.mark.parametrize("resource,file_type", RESOURCES)
def test_delete_site_succeeds_when_it_has_documents(site_id, resource, file_type):
    # DuckDB has no ON DELETE CASCADE, so delete_site clears children in app code.
    client.post(_path(site_id, resource), json={"file_id": _file(file_type)})

    assert client.delete(f"/api/v1/sites/{site_id}").status_code == 204
    assert client.get(f"/api/v1/sites/{site_id}").status_code == 404
