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


class JobSource(BaseModel):
    """The video to read, as it was when the job was created.

    Carried in the message rather than looked up by the worker, for two reasons that
    have nothing to do with saving a round trip — that would be microseconds against a
    multi-minute decode.

    Sources are versioned. Asking site-service for a site's source answers "what is
    active now", so a job enqueued against v3 and consumed after someone attached v4
    would silently read a different video than the one it was created for. A source in
    the message pins the job to the decision that produced it. It also lets a backlog
    drain while site-service is down, which is the point of a queue.

    `key` rather than a download url: a presigned url expires, and one that dies in a
    backlog or midway through a long read fails *after* the worker has started, which
    is the worst place for it. An object key is immutable — a new upload is a new file
    id — so the worker signs its own url whenever it opens the video.
    """

    source_id: str
    # Carried so a job that read the wrong thing is diagnosable rather than merely
    # wrong: the pair identifies exactly which version of the site's source this was.
    version: int
    key: str
    # Copied from the source's probed metadata. The worker needs fps to turn frame
    # indices into times, and neither value is worth a second lookup to obtain.
    fps: float | None = None
    total_frames: int | None = None


class DetectionJob(BaseModel):
    id: str
    site_id: str
    source: JobSource
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
