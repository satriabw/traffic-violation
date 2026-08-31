"""Asking for an explanation, and what happens to the violation afterwards.

The endpoint no longer produces the answer, so these are two separate questions and the
tests keep them apart: what the endpoint accepts and hands over, and what the handler
does once it has it. Both run without a thread — the actor is a dependency, so a test
substitutes one that records instead of one that runs.

No test here reaches llm-service. The explainer is injected the same way the database and
the storage client are, so a fake is a dependency override rather than a patch.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import fail_pending_explanations, record
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

from site_service import service
from site_service.actor import Explain, get_actor
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


class RecordingActor:
    """Takes messages and does nothing with them.

    What the endpoint's own contract is tested against: whether a violation was handed
    over is a different question from what happens to it afterwards, and a fake that ran
    the work would answer both at once and let one hide the other.
    """

    def __init__(self):
        self.sent: list[Explain] = []
        self.status_when_sent: list[str | None] = []

    def send(self, message: Explain) -> None:
        self.sent.append(message)
        # The row as it stood at the moment of handover, for the ordering test below.
        row = _test_con.execute(
            "SELECT status FROM traffic_violations WHERE id = ?", [message.violation_id]
        ).fetchone()
        self.status_when_sent.append(row[0] if row else None)


class InlineActor(RecordingActor):
    """Runs the real handler on send, on the calling thread.

    For the few tests that are about the whole path rather than either half of it. The
    handler is the production one — only the thread is missing, which is the part the
    actor's own tests cover.
    """

    def __init__(self, explainer):
        super().__init__()
        self._explainer = explainer

    def send(self, message: Explain) -> None:
        super().send(message)
        service.perform_explanation(
            _test_con, _storage, self._explainer, message.site_id, message.violation_id
        )


@pytest.fixture
def explainer():
    return FakeExplainer()


@pytest.fixture
def actor():
    return RecordingActor()


@pytest.fixture(autouse=True)
def _overrides(explainer, actor):
    app.dependency_overrides[get_db] = lambda: _test_con
    app.dependency_overrides[get_storage] = lambda: _storage
    app.dependency_overrides[get_probe] = lambda: _probe
    app.dependency_overrides[get_explainer] = lambda: explainer
    # Overridden in every test, so nothing here can start a thread or reach llm-service.
    app.dependency_overrides[get_actor] = lambda: actor
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


def _status(violation_id: str) -> str:
    return _test_con.execute(
        "SELECT status FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()[0]


# --- what the endpoint accepts and hands over ------------------------------------


def test_asking_for_an_explanation_accepts_it_and_returns_immediately(actor, explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    response = _explain(site_id, violation_id)

    assert response.status_code == 202
    assert response.json()["status"] == ViolationStatus.PENDING.value
    assert [message.violation_id for message in actor.sent] == [violation_id]
    # The endpoint itself never calls the model. That is the whole point of the change.
    assert explainer.calls == 0


def test_the_violation_is_pending_before_it_is_handed_over(actor):
    """Order matters more than it looks.

    Send first and a fast actor can finish and write 'explained' before the 'pending'
    write lands, burying a real answer under a status saying it has not started.
    """
    site_id = _site()
    violation_id = _violation(site_id)

    _explain(site_id, violation_id)

    assert actor.status_when_sent == [ViolationStatus.PENDING.value]


def test_a_violation_already_pending_is_not_handed_over_twice(actor):
    # It is in the mailbox or in flight; a second message buys a duplicate model call.
    site_id = _site()
    violation_id = _violation(site_id)

    first = _explain(site_id, violation_id)
    second = _explain(site_id, violation_id)

    assert first.status_code == 202
    # Nothing was accepted the second time, because nothing needed to be.
    assert second.status_code == 200
    assert second.json()["status"] == ViolationStatus.PENDING.value
    assert len(actor.sent) == 1


def test_polling_the_endpoint_never_queues_more_work(actor):
    site_id = _site()
    violation_id = _violation(site_id)

    for _ in range(5):
        _explain(site_id, violation_id)

    assert len(actor.sent) == 1


def test_a_failed_explanation_is_handed_over_again(actor):
    # Asking again is the whole retry mechanism; nothing retries on its own.
    site_id = _site()
    violation_id = _violation(site_id)
    _explain(site_id, violation_id)
    service.set_explanation_status(_test_con, violation_id, ViolationStatus.FAILED)

    response = _explain(site_id, violation_id)

    assert response.status_code == 202
    assert len(actor.sent) == 2


def test_an_explained_violation_is_handed_back_rather_than_re_explained(actor, explainer):
    site_id = _site()
    violation_id = _violation(site_id)
    app.dependency_overrides[get_actor] = lambda: InlineActor(explainer)
    _explain(site_id, violation_id)
    app.dependency_overrides[get_actor] = lambda: actor

    response = _explain(site_id, violation_id)

    assert response.status_code == 200
    assert response.json()["status"] == ViolationStatus.EXPLAINED.value
    assert actor.sent == []
    assert explainer.calls == 1


def test_a_violation_that_does_not_exist_is_a_404_and_is_never_handed_over(actor):
    site_id = _site()

    assert _explain(site_id, "no-such-violation").status_code == 404
    assert actor.sent == []


def test_another_sites_violation_is_a_404_and_is_never_handed_over(actor):
    owner = _site()
    other = _site()
    violation_id = _violation(owner)

    assert _explain(other, violation_id).status_code == 404
    assert actor.sent == []


def test_an_unknown_site_is_a_404_and_is_never_handed_over(actor):
    assert _explain("no-such-site", "whatever").status_code == 404
    assert actor.sent == []


def test_the_endpoint_says_so_when_no_actor_is_running():
    """503, the same answer llm-service gives when its provider is not ready.

    The service is up and the violation is fine; the thing behind this endpoint is not
    running, and the caller should try again rather than be told their request was bad.
    """
    site_id = _site()
    violation_id = _violation(site_id)
    del app.dependency_overrides[get_actor]

    assert _explain(site_id, violation_id).status_code == 503
    # And nothing was accepted — the row is untouched.
    assert _status(violation_id) == ViolationStatus.DETECTED.value


# --- what the handler does with it ------------------------------------------------


def test_the_handler_stores_the_whole_answer(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    service.perform_explanation(_test_con, _storage, explainer, site_id, violation_id)

    detail = client.get(f"{SITES}/{site_id}/violations/{violation_id}").json()
    assert detail["status"] == ViolationStatus.EXPLAINED.value
    assert detail["explanation"] == (
        "A vehicle drove into the junction after the signal had turned red."
    )
    assert detail["severity"] == Severity.MEDIUM.value
    assert detail["explanation_detail"]["evidence_strength"] == "WEAK"
    assert (
        detail["explanation_detail"]["license_plate"]["recoverability"] == "inconclusive"
    )
    assert detail["explanation_detail"]["confidence"] == 0.6


@pytest.mark.parametrize(
    "error",
    [
        # Where a timeout arrives, too: the explainer maps httpx's RequestError onto it.
        ExplainerUnavailable("down"),
        ExplainerRefused("rejected"),
    ],
)
def test_a_failure_lands_on_the_row_rather_than_being_raised(error):
    """There is no caller to raise to.

    This runs on a thread nobody is waiting on, so an exception that escaped would be a
    log line and a violation stuck saying 'pending' for good.
    """
    site_id = _site()
    violation_id = _violation(site_id)
    service.request_explanation(_test_con, _storage, site_id, violation_id)

    service.perform_explanation(
        _test_con, _storage, FakeExplainer(error=error), site_id, violation_id
    )

    assert _status(violation_id) == ViolationStatus.FAILED.value


def test_a_failure_does_not_discard_an_explanation_already_on_the_row(explainer):
    # A violation explained once and failed on a later attempt keeps its first answer.
    site_id = _site()
    violation_id = _violation(site_id)
    service.perform_explanation(_test_con, _storage, explainer, site_id, violation_id)

    service.perform_explanation(
        _test_con,
        _storage,
        FakeExplainer(error=ExplainerUnavailable("down")),
        site_id,
        violation_id,
    )

    status, explanation = _test_con.execute(
        "SELECT status, explanation FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()
    assert status == ViolationStatus.FAILED.value
    assert explanation is not None


def test_a_violation_deleted_before_it_is_handled_writes_nothing(explainer):
    site_id = _site()

    service.perform_explanation(
        _test_con, _storage, explainer, site_id, "no-such-violation"
    )

    assert explainer.calls == 0


def test_the_explainer_is_told_when_there_is_no_calibration_to_trust(explainer):
    # The prompt withholds the motion data on this, which is the single decision the
    # whole prompt study turned on.
    site_id = _site()
    violation_id = _violation(site_id, calibration_id=None)

    service.perform_explanation(_test_con, _storage, explainer, site_id, violation_id)

    assert explainer.requests[0].calibration_id is None


def test_the_explainer_gets_the_scene_the_detector_recorded(explainer):
    site_id = _site()
    violation_id = _violation(site_id)

    service.perform_explanation(_test_con, _storage, explainer, site_id, violation_id)

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

    service.perform_explanation(_test_con, _storage, explainer, site_id, violation_id)

    payload = explainer.requests[0].model_dump_json()
    assert violation_id not in payload
    assert site_id not in payload


# --- the whole path, and what a restart does to it --------------------------------


def test_an_explanation_shows_up_on_the_detail_read_and_the_list(explainer):
    site_id = _site()
    violation_id = _violation(site_id)
    app.dependency_overrides[get_actor] = lambda: InlineActor(explainer)

    response = _explain(site_id, violation_id)

    assert response.status_code == 202
    detail = client.get(f"{SITES}/{site_id}/violations/{violation_id}").json()
    assert detail["status"] == ViolationStatus.EXPLAINED.value

    listed = client.get(f"{SITES}/{site_id}/violations").json()["items"][0]
    # The flat columns are what the list renders — it never parses the blob.
    assert listed["severity"] == Severity.MEDIUM.value
    assert listed["status"] == ViolationStatus.EXPLAINED.value
    assert listed["explanation_detail"] is None


def test_a_violation_left_pending_by_a_restart_is_failed_not_stranded(actor):
    """What startup does, and why the mailbox living in memory is survivable.

    The message died with the process. Without this the row promises an answer nothing
    is going to produce, and a client polls it forever.
    """
    site_id = _site()
    violation_id = _violation(site_id)
    _explain(site_id, violation_id)
    assert _status(violation_id) == ViolationStatus.PENDING.value

    assert fail_pending_explanations(_test_con) == 1

    assert _status(violation_id) == ViolationStatus.FAILED.value


def test_startup_leaves_alone_anything_that_was_not_pending(explainer):
    site_id = _site()
    explained = _violation(site_id)
    detected = _violation(site_id)
    service.perform_explanation(_test_con, _storage, explainer, site_id, explained)

    assert fail_pending_explanations(_test_con) == 0

    assert _status(explained) == ViolationStatus.EXPLAINED.value
    assert _status(detected) == ViolationStatus.DETECTED.value
