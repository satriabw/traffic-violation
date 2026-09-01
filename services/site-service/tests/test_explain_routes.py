"""The explain endpoint: the cache, the errors, and what reaches the explainer.

No test here reaches llm-service. The explainer is injected the same way the database
and the storage client are, so a fake is a dependency override rather than a patch —
which is also what makes "did it call?" a thing a test can simply ask.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import record
from shared.models.detection import ViolationType
from shared.models.explanation import ExplainResponse
from shared.models.violation import (
    EvidenceStrength,
    LicensePlateAssessment,
    PlateRecoverability,
    Severity,
    TrackSummary,
    ViolationCreate,
    ViolationExplanation,
    ViolationMetadata,
    ViolationStatus,
)

from site_service.llm import ExplainerRefused, ExplainerUnavailable, get_explainer
from site_service.main import app, get_db
from site_service.storage import get_storage
from site_service.video import get_probe
from tests.test_file_service import FakeStorage
from tests.test_source_routes import FakeProbe

SITES = "/api/v1/sites"
DETECTED_AT = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)

_test_con = get_connection(":memory:")
init_db(_test_con)
_storage = FakeStorage(existing_size=64)
_probe = FakeProbe()

client = TestClient(app, raise_server_exceptions=False)


class FakeExplainer:
    """Answers without a network, and remembers what it was asked."""

    def __init__(self, error=None):
        self.error = error
        self.requests = []
        self.answer = ViolationExplanation(
            explanation="A vehicle drove into the junction after the signal had turned red.",
            severity=Severity.MEDIUM,
            severity_basis=["other traffic was moving through at the time"],
            evidence_strength=EvidenceStrength.WEAK,
            evidence_basis=["the record cannot confirm the signal was red; the footage can"],
            license_plate=LicensePlateAssessment(
                recoverability=PlateRecoverability.INCONCLUSIVE,
                reasoning="The vehicle stays distant and there is no plate recognition here.",
            ),
            observations=["Several objects counted as vehicles never move at all."],
            evidence_concerns=["Disregard any speed shown — the camera calibration is faulty."],
            confidence=0.6,
        )

    def explain(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return ExplainResponse(explanation=self.answer, model="fake-model-1")

    @property
    def calls(self):
        return len(self.requests)


@pytest.fixture
def explainer():
    return FakeExplainer()


@pytest.fixture(autouse=True)
def _overrides(explainer):
    app.dependency_overrides[get_db] = lambda: _test_con
    app.dependency_overrides[get_storage] = lambda: _storage
    app.dependency_overrides[get_probe] = lambda: _probe
    app.dependency_overrides[get_explainer] = lambda: explainer
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


def _violation(site_id: str, calibration_id=None) -> str:
    return record(
        _test_con,
        ViolationCreate(
            site_id=site_id,
            source_id=f"src-{site_id}",
            frame_index=912,
            calibration_id=calibration_id,
            type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY,
            detected_at=DETECTED_AT,
            metadata=ViolationMetadata(
                vehicles=[TrackSummary(track_id=19, frame_idxs=[910, 912])],
                pedestrians=[TrackSummary(track_id=7, frame_idxs=[911])],
                violator_track_id=19,
            ),
        ),
    )


def _explain(site_id: str, violation_id: str):
    return client.post(f"{SITES}/{site_id}/violations/{violation_id}/explain")


def test_explaining_an_unexplained_violation_calls_the_explainer_and_stores_it(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    response = _explain(site_id, violation_id)

    assert response.status_code == 200
    assert explainer.calls == 1
    body = response.json()
    assert body["explanation"] == (
        "A vehicle drove into the junction after the signal had turned red."
    )
    assert body["severity"] == Severity.MEDIUM.value
    assert body["status"] == ViolationStatus.EXPLAINED.value
    # The whole answer, not just the two flat fields.
    assert body["explanation_detail"]["evidence_concerns"] == [
        "Disregard any speed shown — the camera calibration is faulty."
    ]
    # The clerk-facing additions survive the round trip through the stored blob.
    assert body["explanation_detail"]["evidence_strength"] == "WEAK"
    assert (
        body["explanation_detail"]["license_plate"]["recoverability"]
        == "inconclusive"
    )


def test_a_second_request_reads_the_database_instead_of_calling_again(explainer):
    # The reason the endpoint is safe to call on every page load.
    site_id = _site()
    violation_id = _violation(site_id)

    first = _explain(site_id, violation_id).json()
    second = _explain(site_id, violation_id).json()

    assert explainer.calls == 1
    assert second["explanation"] == first["explanation"]
    assert second["explanation_detail"] == first["explanation_detail"]


def test_many_requests_still_only_ever_cost_one_call(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    for _ in range(5):
        assert _explain(site_id, violation_id).status_code == 200

    assert explainer.calls == 1


def test_a_violation_that_does_not_exist_is_a_404_and_costs_nothing(explainer):
    site_id = _site()

    assert _explain(site_id, "no-such-violation").status_code == 404
    assert explainer.calls == 0


def test_another_sites_violation_is_a_404_and_costs_nothing(explainer):
    owner = _site()
    other = _site()
    violation_id = _violation(owner)

    assert _explain(other, violation_id).status_code == 404
    assert explainer.calls == 0


def test_an_unknown_site_is_a_404_and_costs_nothing(explainer):
    assert _explain("no-such-site", "whatever").status_code == 404
    assert explainer.calls == 0


def test_the_explainer_is_told_when_there_is_no_calibration_to_trust(explainer):
    # The prompt withholds the motion data on this, which is the single decision the
    # whole prompt study turned on.
    site_id = _site()
    violation_id = _violation(site_id, calibration_id=None)

    _explain(site_id, violation_id)

    assert explainer.requests[0].calibration_id is None


def test_the_explainer_gets_the_scene_the_detector_recorded(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    _explain(site_id, violation_id)

    request = explainer.requests[0]
    assert request.site_name == "Junction 5"
    assert request.frame_index == 912
    assert request.metadata.violator_track_id == 19
    assert len(request.metadata.pedestrians) == 1


def test_the_explainer_is_never_told_which_violation_it_is(explainer):
    # llm-service does not read or write the database, and keeping the identifiers out
    # of the request is what makes that true rather than merely intended.
    site_id = _site()
    violation_id = _violation(site_id)

    _explain(site_id, violation_id)

    assert violation_id not in explainer.requests[0].model_dump_json()
    assert site_id not in explainer.requests[0].model_dump_json()


@pytest.mark.parametrize(
    "error, expected",
    [
        (ExplainerUnavailable("down"), 503),
        (ExplainerRefused("rejected"), 502),
    ],
)
def test_explainer_failures_keep_their_meaning_on_the_way_out(error, expected):
    site_id = _site()
    violation_id = _violation(site_id)
    app.dependency_overrides[get_explainer] = lambda: FakeExplainer(error=error)

    assert _explain(site_id, violation_id).status_code == expected


def test_a_failed_explanation_leaves_the_violation_unexplained():
    # So the next request tries again rather than caching a failure.
    site_id = _site()
    violation_id = _violation(site_id)
    app.dependency_overrides[get_explainer] = lambda: FakeExplainer(
        error=ExplainerUnavailable("down")
    )

    _explain(site_id, violation_id)

    row = _test_con.execute(
        "SELECT status, explanation FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()
    assert row == (ViolationStatus.DETECTED.value, None)


def test_an_explanation_shows_up_on_the_detail_read_and_the_list(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    _explain(site_id, violation_id)

    detail = client.get(f"{SITES}/{site_id}/violations/{violation_id}").json()
    assert detail["explanation_detail"]["confidence"] == 0.6

    listed = client.get(f"{SITES}/{site_id}/violations").json()["items"][0]
    # The flat columns are what the list renders — it never parses the blob.
    assert listed["severity"] == Severity.MEDIUM.value
    assert listed["explanation_detail"] is None
