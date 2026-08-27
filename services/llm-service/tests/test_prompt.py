from datetime import datetime, timezone

from shared.models.detection import ViolationType
from shared.models.explanation import ExplainRequest
from shared.models.violation import TrackSummary, ViolationMetadata

from llm_service.prompt import build_prompt


def _request(**overrides):
    return ExplainRequest(
        **{
            "violation_type": ViolationType.RED_LIGHT_RUNNING,
            "detected_at": datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
            "site_name": "Junction 5",
            "frame_index": 159,
            **overrides,
        }
    )


def test_a_violation_with_no_track_record_says_so_rather_than_inventing_one():
    prompt = build_prompt(_request(metadata=None))

    assert "Track record: MISSING" in prompt


def test_a_missing_configuration_is_named_rather_than_omitted():
    # An absent configuration changes what the explanation can claim, so it has to
    # reach the model as a fact rather than as silence.
    prompt = build_prompt(_request(configuration=None))

    assert "MISSING" in prompt


def test_pedestrians_on_the_scene_reach_the_model():
    # The severity rubric's HIGH band turns on this, so it is the one count that
    # cannot be summarised away.
    prompt = build_prompt(
        _request(
            metadata=ViolationMetadata(
                vehicles=[TrackSummary(track_id=1, frame_idxs=[1, 2])],
                pedestrians=[TrackSummary(track_id=7, frame_idxs=[1, 2, 3])],
                violator_track_id=1,
            )
        )
    )

    assert "Pedestrians on the scene: 1" in prompt
    assert "pedestrian track 7: frames 1-3, 3 samples" in prompt


def test_the_whole_track_record_is_summarised_not_dumped():
    # ~13.5KB per track of samples says nothing a reader of the explanation needs,
    # and paying for it per violation is the cost the metadata table exists to avoid.
    prompt = build_prompt(
        _request(
            metadata=ViolationMetadata(
                vehicles=[
                    TrackSummary(
                        track_id=19,
                        frame_idxs=list(range(45, 160)),
                        speed=[1.0] * 115,
                        bboxes=[(0.0, 0.0, 1.0, 1.0)] * 115,
                    )
                ],
                violator_track_id=19,
            )
        )
    )

    assert "vehicle track 19: frames 45-159, 115 samples" in prompt
    assert "0.0, 0.0, 1.0, 1.0" not in prompt


def test_an_uncalibrated_violation_is_told_why_the_fallbacks_do_not_work():
    # Naming the failure without naming its mechanism produced a partial correction
    # that read as a complete one, so both routes are spelled out.
    prompt = build_prompt(_request(calibration_id=None))

    assert "same mapping as the speeds" in prompt
    assert "along the camera's axis" in prompt


def test_no_severity_band_can_be_reached_by_a_speed():
    # Every over-graded severity in the study came from a run that believed the
    # telemetry, and a band phrased as "materially above the limit" invites exactly
    # that. The bands themselves are checked, not the preamble above them — that one
    # says "do not grade on speed", which is the point rather than a violation of it.
    prompt = build_prompt(_request(calibration_id="cal-1"))

    bands = prompt[prompt.index("HIGH   -"):prompt.index("State in severity_basis")]
    assert "speed" not in bands.lower()
    assert "do not grade on speed" in prompt
