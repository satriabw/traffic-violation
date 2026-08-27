"""The wire path for the violation list. What the setup filter *means* is tested at
the service layer in test_violation_service.py; these are the things only a request
can show — the route, the 404, the query bounds, and the json a client actually gets."""

import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import record, set_evidence, set_explanation
from shared.models.detection import ViolationType
from shared.models.violation import (
    EvidenceStatus,
    Severity,
    TrackSummary,
    ViolationCreate,
    ViolationExplanation,
    ViolationMetadata,
    ViolationStatus,
)

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


# --- the detail endpoint ---------------------------------------------------


def _detail(site_id: str, violation_id: str):
    return client.get(f"{SITES}/{site_id}/violations/{violation_id}")


def _with_tracks(site_id: str) -> str:
    return record(
        _test_con,
        ViolationCreate(
            site_id=site_id,
            source_id=f"src-{site_id}",
            frame_index=912,
            type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY,
            detected_at=DETECTED_AT,
            metadata=ViolationMetadata(
                vehicles=[TrackSummary(track_id=19, frame_idxs=[910, 911, 912])],
                pedestrians=[TrackSummary(track_id=7, frame_idxs=[911, 912])],
                violator_track_id=19,
            ),
        ),
    )


def test_the_detail_view_carries_the_blob_the_list_does_not():
    site_id = _site()
    violation_id = _with_tracks(site_id)

    response = _detail(site_id, violation_id)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["violator_track_id"] == 19
    assert body["metadata"]["pedestrians"][0]["track_id"] == 7
    # The same violation through the list carries none of it.
    assert _get(site_id).json()["items"][0]["metadata"] is None


def test_a_violation_that_does_not_exist_is_a_404():
    site_id = _site()

    assert _detail(site_id, "no-such-violation").status_code == 404


def test_a_violation_cannot_be_read_through_another_sites_url():
    # The id is a uuid and the route is nested, so without the site check a caller
    # holding an id from one site could read it under another's — and every reader
    # trusting the path to mean what it says would be wrong.
    owner = _site()
    other = _site()
    violation_id = _with_tracks(owner)

    assert _detail(other, violation_id).status_code == 404


def test_a_violation_under_the_wrong_site_is_not_distinguishable_from_a_missing_one():
    # Telling the two apart would confirm the id is real and belongs to somebody else.
    owner = _site()
    other = _site()
    violation_id = _with_tracks(owner)

    assert _detail(other, violation_id).json() == _detail(other, "nope").json()


def test_the_detail_view_of_an_unknown_site_is_a_404():
    assert _detail("no-such-site", "whatever").status_code == 404


def test_an_unexplained_violation_reads_back_with_no_explanation():
    site_id = _site()
    violation_id = _with_tracks(site_id)

    body = _detail(site_id, violation_id).json()

    assert body["status"] == ViolationStatus.DETECTED.value
    assert body["explanation"] is None
    assert body["severity"] is None
    assert body["explanation_detail"] is None


def test_an_explained_violation_carries_the_whole_answer_back():
    site_id = _site()
    violation_id = _with_tracks(site_id)
    explanation = ViolationExplanation(
        explanation="Entered against a red signal.",
        severity=Severity.MEDIUM,
        severity_basis=["one pedestrian track on the scene"],
        observations=["two tracks recorded"],
        evidence_concerns=["speeds uncalibrated"],
        confidence=0.6,
    )
    set_explanation(
        _test_con,
        violation_id,
        explanation.explanation,
        explanation.severity.value,
        explanation.model_dump_json(),
    )

    body = _detail(site_id, violation_id).json()

    assert body["status"] == ViolationStatus.EXPLAINED.value
    assert body["explanation"] == "Entered against a red signal."
    assert body["severity"] == "MEDIUM"
    # The field the flat columns cannot hold, parsed back into the model rather than
    # handed over as text.
    assert body["explanation_detail"]["evidence_concerns"] == ["speeds uncalibrated"]
    assert body["explanation_detail"]["severity_basis"] == ["one pedestrian track on the scene"]


def test_reading_a_violation_never_explains_it():
    # A GET that quietly called a model would spend money and write on a prefetch.
    site_id = _site()
    violation_id = _with_tracks(site_id)

    _detail(site_id, violation_id)
    _detail(site_id, violation_id)

    status = _test_con.execute(
        "SELECT status, explanation FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()
    assert status == (ViolationStatus.DETECTED.value, None)


def test_the_detail_view_signs_the_evidence_the_same_way_the_list_does():
    site_id = _site()
    violation_id = _with_tracks(site_id)
    set_evidence(
        _test_con, violation_id, EvidenceStatus.READY, "thumb/a.jpg", "clip/a.mp4"
    )

    body = _detail(site_id, violation_id).json()

    assert body["thumbnail_url"] is not None
    assert body["clip_url"] is not None
    assert body["evidence_status"] == EvidenceStatus.READY.value


def test_a_violation_that_is_returned_by_the_list_is_readable_by_id():
    # The detail view applies no setup filter, so anything the list shows must resolve.
    site_id = _site()
    violation_id = _with_tracks(site_id)

    listed = _get(site_id).json()["items"][0]["id"]

    assert listed == violation_id
    assert _detail(site_id, listed).status_code == 200
