"""The detection job — the one thing site-service and detection-worker both know.

It travels as json over a queue rather than as a row both sides can read, so this
module is the contract between two processes. Anything the worker needs in order to
start has to be in here; anything it can look up for itself should not be.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ViolationType(str, Enum):
    RED_LIGHT_RUNNING = "red_light_running"
    PEDESTRIAN_RIGHT_OF_WAY = "pedestrian_right_of_way"


class FrameRange(BaseModel):
    """Half-open [start, end), in frame indices — the same units the probe reports.

    Frames, not seconds: variable-rate footage makes a time offset ambiguous, and
    SourceMetadata already keeps fps and nominal_fps apart for exactly that reason.
    """

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _range_is_not_empty(self):
        # Enforced here rather than by whoever happens to read the message: a job
        # covering no frames is work the worker cannot do, and it should be
        # unrepresentable rather than caught four processes later.
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class DetectionJob(BaseModel):
    id: str
    site_id: str
    frame_range: FrameRange
    # At least one: a job asking for no violation types is a no-op nobody wants queued.
    types: list[ViolationType] = Field(min_length=1)


class DetectionRequest(BaseModel):
    """The POST body. Everything else about the job is derived server-side."""

    # Strict, so a client sending frame_range learns it comes from the source's probed
    # metadata rather than having it silently ignored.
    model_config = ConfigDict(extra="forbid")

    # None means "every type we know about" — a caller who wants everything checked
    # should not have to enumerate it, and the list grows without breaking them.
    types: list[ViolationType] | None = Field(default=None, min_length=1)
