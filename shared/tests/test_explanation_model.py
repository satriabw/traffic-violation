"""The contract on what an explainer returns, and on what was already stored.

`explanation_json` is parsed straight back into `ViolationExplanation` on every detail
read, so this model is not free to grow a required field: there are rows in the real
database written under the previous shape, and a required addition turns their detail
endpoint into a 500. The defaults are load-bearing rather than decorative, which is what
these pin.
"""

import json

from shared.models.violation import (
    EvidenceStrength,
    LicensePlateAssessment,
    PlateRecoverability,
    Severity,
    ViolationExplanation,
)

# An explanation exactly as it was stored before evidence strength and the plate
# assessment existed. Not a fixture built from the current model — the whole point is
# that it was written by code that had never heard of the new fields.
LEGACY_JSON = json.dumps(
    {
        "flag_sustained": True,
        "explanation": "Entered against a red signal.",
        "severity": "MEDIUM",
        "severity_basis": ["two other vehicle tracks present"],
        "observations": ["three tracks on the scene"],
        "evidence_concerns": ["speeds uncalibrated"],
        "confidence": 0.6,
    }
)


def test_an_explanation_stored_before_these_fields_existed_still_parses():
    explanation = ViolationExplanation.model_validate_json(LEGACY_JSON)

    assert explanation.severity is Severity.MEDIUM
    assert explanation.explanation == "Entered against a red signal."


def test_the_new_fields_read_as_absent_rather_than_as_a_verdict():
    """None means nobody was asked, not that somebody looked and found nothing.

    A default of WEAK would have been the tempting alternative and is a different claim:
    it says the record was examined and came up short, about a violation explained before
    anything examined it.
    """
    explanation = ViolationExplanation.model_validate_json(LEGACY_JSON)

    assert explanation.evidence_strength is None
    assert explanation.license_plate is None
    assert explanation.evidence_basis == []


def test_a_current_explanation_round_trips_through_the_stored_column():
    # What set_explanation writes and _row_to_violation_detail reads back, in one step.
    answer = ViolationExplanation(
        explanation="A vehicle drove into the junction after the signal had turned red.",
        severity=Severity.LOW,
        severity_basis=["nobody else was there"],
        evidence_strength=EvidenceStrength.WEAK,
        evidence_basis=["the record cannot confirm the signal was red"],
        license_plate=LicensePlateAssessment(
            recoverability=PlateRecoverability.MANUAL_READ,
            reasoning="The vehicle passes close to the camera and footage was kept.",
        ),
        observations=["Several objects counted as vehicles never move at all."],
        evidence_concerns=["Disregard any speed shown — the calibration is faulty."],
        confidence=0.55,
    )

    restored = ViolationExplanation.model_validate_json(answer.model_dump_json())

    assert restored.evidence_strength is EvidenceStrength.WEAK
    assert restored.license_plate is not None
    assert restored.license_plate.recoverability is PlateRecoverability.MANUAL_READ


def test_severity_and_evidence_strength_are_independent():
    """A serious event can be thinly evidenced, and a trivial one established beyond doubt.

    Nothing in the model couples them, and this is the test that says so on purpose —
    they are the two axes a reviewer is most likely to conflate.
    """
    answer = ViolationExplanation(
        explanation="A vehicle drove into a crossing somebody was already in.",
        severity=Severity.HIGH,
        severity_basis=["somebody was on foot in the crossing at the time"],
        evidence_strength=EvidenceStrength.WEAK,
        evidence_basis=["the record does not say which crossing either was in"],
        confidence=0.4,
    )

    assert answer.severity is Severity.HIGH
    assert answer.evidence_strength is EvidenceStrength.WEAK
