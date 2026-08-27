"""The wire path for the violation list. What the setup filter *means* is tested at
the service layer in test_violation_service.py; these are the things only a request
can show — the route, the 404, the query bounds, and the json a client actually gets."""

import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import record, set_evidence
from shared.models.detection import ViolationType
from shared.models.violation import EvidenceStatus, ViolationCreate

from site_service.main import app, get_db
from site_service.storage import get_storage
from site_service.video import get_probe
from tests.test_file_service import FakeStorage
from tests.test_source_routes import FakeProbe

from datetime import datetime, timezone

SITES = "/api/v1/sites"
DETECTED_AT = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=64)
_probe = FakeProbe()

client = TestClient(app)


@pytest.fixture(autouse=True)
def _dependency_overrides():
    app.dependency_overrides[get_db] = lambda: _test_con
    app.dependency_overrides[get_storage] = lambda: _storage
    app.dependency_overrides[get_probe] = lambda: _probe
    _test_con.execute("DELETE FROM traffic_violations")
    yield
    app.dependency_overrides.clear()


def _site() -> str:
    site_id = client.post(SITES, json={"name": "Junction 5"}).json()["id"]
    _test_con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES (?, 'a.mp4', 'video/f/a.mp4', 'video', 'uploaded')",
        [f"file-{site_id}"],
    )
    _test_con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id)"
        " VALUES (?, ?, 1, 'video', ?)",
        [f"src-{site_id}", site_id, f"file-{site_id}"],
    )
    return site_id


def _violation(site_id: str) -> str:
    return record(
        _test_con,
        ViolationCreate(
            site_id=site_id,
            source_id=f"src-{site_id}",
            frame_index=912,
            type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY,
            detected_at=DETECTED_AT,
        ),
    )


def _get(site_id: str, **params):
    return client.get(f"{SITES}/{site_id}/violations", params=params)


def test_a_sites_violations_are_served_under_the_site():
    site_id = _site()
    violation_id = _violation(site_id)

    response = _get(site_id)

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]] == [violation_id]
    assert (body["total"], body["limit"], body["offset"]) == (1, 20, 0)


def test_an_unknown_site_is_a_404_not_an_empty_page():
    # An empty list would tell a client its site has no violations, which is a
    # different fact from the site not existing and sends it looking in the wrong place.
    response = _get("nope")

    assert response.status_code == 404
    assert response.json()["detail"] == "Site not found"


def test_a_site_with_no_violations_is_an_empty_page():
    response = _get(_site())

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_the_evidence_comes_back_as_links_a_reviewer_can_open():
    site_id = _site()
    violation_id = _violation(site_id)
    set_evidence(
        _test_con,
        violation_id,
        EvidenceStatus.READY,
        thumbnail_key=f"evidence/{violation_id}/thumbnail.jpg",
        clip_key=f"evidence/{violation_id}/clip.mp4",
    )

    (item,) = _get(site_id).json()["items"]

    assert item["thumbnail_url"] == (
        f"https://r2.test/get/evidence/{violation_id}/thumbnail.jpg"
    )
    assert item["clip_url"] == f"https://r2.test/get/evidence/{violation_id}/clip.mp4"
    assert item["evidence_status"] == "ready"


def test_a_violation_still_being_cut_has_no_links_yet():
    site_id = _site()
    set_evidence(_test_con, _violation(site_id), EvidenceStatus.PENDING)

    (item,) = _get(site_id).json()["items"]

    assert item["thumbnail_url"] is None and item["clip_url"] is None
    assert item["evidence_status"] == "pending"


def test_the_page_never_carries_the_metadata_blob():
    site_id = _site()
    _violation(site_id)

    (item,) = _get(site_id).json()["items"]

    # The whole reason violation_metadata is a separate table. A client that wants the
    # boxes asks for one violation, not for a page of them.
    assert item["metadata"] is None


def test_the_page_is_the_window_the_client_asked_for():
    site_id = _site()
    for _ in range(3):
        _violation(site_id)

    body = _get(site_id, limit=2, offset=1).json()

    assert (len(body["items"]), body["total"]) == (2, 3)
    assert (body["limit"], body["offset"]) == (2, 1)


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 101}, {"limit": -1}, {"offset": -1}]
)
def test_a_page_outside_the_bounds_is_rejected(params):
    # An unbounded limit is a client able to ask for every violation a site ever
    # recorded in one response, blob-free or not.
    assert _get(_site(), **params).status_code == 422
