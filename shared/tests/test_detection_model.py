import pytest
from pydantic import ValidationError

from shared.models.detection import (
    DetectionJob,
    DetectionRequest,
    FrameRange,
    ViolationType,
)


def _job(**overrides) -> DetectionJob:
    return DetectionJob(
        **{
            "id": "job-1",
            "site_id": "site-1",
            "frame_range": FrameRange(start=0, end=27000),
            "types": [ViolationType.RED_LIGHT_RUNNING],
            **overrides,
        }
    )


def test_job_survives_a_json_round_trip():
    # The queue carries json, so this is the only property that actually matters:
    # what the worker parses must equal what the service pushed.
    job = _job(types=list(ViolationType))

    assert DetectionJob.model_validate_json(job.model_dump_json()) == job


def test_frame_range_rejects_an_empty_or_backwards_range():
    with pytest.raises(ValidationError):
        FrameRange(start=100, end=100)
    with pytest.raises(ValidationError):
        FrameRange(start=100, end=99)


def test_frame_range_rejects_a_negative_start():
    with pytest.raises(ValidationError):
        FrameRange(start=-1, end=10)


def test_job_rejects_an_empty_type_list():
    # A job asking for no violation types is work nobody can do.
    with pytest.raises(ValidationError):
        _job(types=[])


def test_job_rejects_an_unknown_violation_type():
    with pytest.raises(ValidationError):
        _job(types=["jaywalking"])


def test_request_defaults_types_to_unset():
    assert DetectionRequest().types is None


def test_request_rejects_unknown_fields():
    # Same strictness as SiteCreate: a client sending frame_range here should learn
    # it is derived from the source, not silently ignored.
    with pytest.raises(ValidationError):
        DetectionRequest(frame_range={"start": 0, "end": 10})
