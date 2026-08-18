from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from site_service.main import app, get_db

# Spelled out rather than imported from shared.config so the tests pin the wire
# path clients actually call, and a change to the prefix has to be deliberate.
SITES = "/api/v1/sites"

_test_con = get_connection(":memory:")
init_db(_test_con)


def override_get_db():
    yield _test_con


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_is_served_outside_the_version_prefix():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unprefixed_resource_path_is_not_served():
    assert client.get("/sites").status_code == 404


def test_create_site_returns_201():
    response = client.post(
        SITES, json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "video"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Main St"
    assert body["status"] == "created"


def test_create_site_returns_422_on_invalid_mode():
    response = client.post(
        SITES, json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "bogus"}
    )

    assert response.status_code == 422


def test_get_site_returns_200_when_found():
    created = client.post(
        SITES, json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "video"}
    ).json()

    response = client.get(f"{SITES}/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_site_returns_404_when_missing():
    response = client.get(f"{SITES}/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_list_sites_returns_envelope():
    client.post(SITES, json={"name": "A", "url": "s3://a", "mode": "video"})
    client.post(SITES, json={"name": "B", "url": "s3://b", "mode": "stream"})

    response = client.get(SITES)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    assert "items" in body


def test_list_sites_filters_by_mode():
    client.post(SITES, json={"name": "OnlyStream", "url": "s3://s", "mode": "stream"})

    response = client.get(SITES, params={"mode": "stream"})

    assert response.status_code == 200
    body = response.json()
    assert all(item["mode"] == "stream" for item in body["items"])


def test_delete_site_returns_204_when_found():
    created = client.post(
        SITES, json={"name": "ToDelete", "url": "s3://x", "mode": "video"}
    ).json()

    response = client.delete(f"{SITES}/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"{SITES}/{created['id']}").status_code == 404


def test_delete_site_returns_404_when_missing():
    response = client.delete(f"{SITES}/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def _create_site(name="CalSite"):
    return client.post(SITES, json={"name": name, "url": "s3://v", "mode": "video"}).json()


def test_unprefixed_calibrations_path_is_not_served():
    site = _create_site()

    assert client.get(f"/sites/{site['id']}/calibrations").status_code == 404


def test_create_calibration_returns_201_with_version_one():
    site = _create_site()

    response = client.post(
        f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v1.json"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["version"] == 1
    assert body["site_id"] == site["id"]
    assert body["url"] == "s3://cal/v1.json"


def test_create_calibration_returns_404_for_unknown_site():
    response = client.post(f"{SITES}/does-not-exist/calibrations", json={"url": "s3://a"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_create_calibration_returns_422_without_url():
    site = _create_site()

    response = client.post(f"{SITES}/{site['id']}/calibrations", json={})

    assert response.status_code == 422


def test_get_calibrations_returns_the_active_version():
    site = _create_site()
    client.post(f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v1.json"})
    client.post(f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v2.json"})

    response = client.get(f"{SITES}/{site['id']}/calibrations")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 2
    assert body["url"] == "s3://cal/v2.json"


def test_get_calibrations_returns_404_when_site_has_none():
    site = _create_site()

    response = client.get(f"{SITES}/{site['id']}/calibrations")

    assert response.status_code == 404
    assert response.json() == {"detail": "Calibration not found"}


def test_get_calibrations_returns_404_for_unknown_site():
    response = client.get(f"{SITES}/does-not-exist/calibrations")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_get_calibration_by_id_returns_an_older_version():
    site = _create_site()
    first = client.post(
        f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v1.json"}
    ).json()
    client.post(f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v2.json"})

    response = client.get(f"{SITES}/{site['id']}/calibrations/{first['id']}")

    assert response.status_code == 200
    assert response.json()["version"] == 1


def test_get_calibration_by_id_returns_404_for_another_sites_calibration():
    owner = _create_site("owner")
    stranger = _create_site("stranger")
    calibration = client.post(
        f"{SITES}/{owner['id']}/calibrations", json={"url": "s3://cal/v1.json"}
    ).json()

    response = client.get(f"{SITES}/{stranger['id']}/calibrations/{calibration['id']}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Calibration not found"}


def test_delete_site_with_calibrations_returns_204():
    site = _create_site()
    client.post(f"{SITES}/{site['id']}/calibrations", json={"url": "s3://cal/v1.json"})

    response = client.delete(f"{SITES}/{site['id']}")

    assert response.status_code == 204
    assert client.get(f"{SITES}/{site['id']}").status_code == 404
