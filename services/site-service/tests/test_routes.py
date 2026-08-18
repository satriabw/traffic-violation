from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db

from site_service.main import app, get_db

_test_con = get_connection(":memory:")
init_db(_test_con)


def override_get_db():
    yield _test_con


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_create_site_returns_201():
    response = client.post(
        "/site", json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "video"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Main St"
    assert body["status"] == "created"


def test_create_site_returns_422_on_invalid_mode():
    response = client.post(
        "/site", json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "bogus"}
    )

    assert response.status_code == 422


def test_get_site_returns_200_when_found():
    created = client.post(
        "/site", json={"name": "Main St", "url": "s3://bucket/a.mp4", "mode": "video"}
    ).json()

    response = client.get(f"/site/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_site_returns_404_when_missing():
    response = client.get("/site/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}


def test_list_sites_returns_envelope():
    client.post("/site", json={"name": "A", "url": "s3://a", "mode": "video"})
    client.post("/site", json={"name": "B", "url": "s3://b", "mode": "stream"})

    response = client.get("/site")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    assert "items" in body


def test_list_sites_filters_by_mode():
    client.post("/site", json={"name": "OnlyStream", "url": "s3://s", "mode": "stream"})

    response = client.get("/site", params={"mode": "stream"})

    assert response.status_code == 200
    body = response.json()
    assert all(item["mode"] == "stream" for item in body["items"])


def test_delete_site_returns_204_when_found():
    created = client.post(
        "/site", json={"name": "ToDelete", "url": "s3://x", "mode": "video"}
    ).json()

    response = client.delete(f"/site/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"/site/{created['id']}").status_code == 404


def test_delete_site_returns_404_when_missing():
    response = client.delete("/site/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Site not found"}
