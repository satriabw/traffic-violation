"""What the worker records when a rule fires, and what a client reads back.

The row and the blob are modelled separately because they are read separately: the
list endpoint wants a page of violations and none of their trajectories, and only the
detail view pays for the metadata. Splitting them here mirrors the two tables.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Reused rather than redefined. A violation's type and the types a detection job was
# asked to look for are the same vocabulary, and two enums would drift.
from shared.models.detection import ViolationType


class ViolationStatus(str, Enum):
    # Written by the worker. Everything the detector knows is already in the row.
    DETECTED = "detected"
    # An LLM has since filled in explanation and severity. Set at read time, on
    # demand, by whoever asked for the detail view — never by the worker.
    EXPLAINED = "explained"


class TrackSummary(BaseModel):
    """One object's path through the window that produced the violation.

    The four lists are parallel — index i of each describes the same frame — which is
    what makes a trajectory point matchable to the box it came from. `track_id` is the
    tracker's id, and is only meaningful within the job that assigned it.
    """

    track_id: int
    # Image coordinates unless a calibration was applied, in which case ground-plane
    # metres. Which one it is follows from whether the job carried a calibration.
    trajectory: list[tuple[float, float]] = Field(default_factory=list)
    speed: list[float] = Field(default_factory=list)
    frame_idxs: list[int] = Field(default_factory=list)
    # (x1, y1, x2, y2), the same convention supervision uses.
    bboxes: list[tuple[float, float, float, float]] = Field(default_factory=list)


class ViolationMetadata(BaseModel):
    """The json_blob, typed. Shape is the LLD's."""

    vehicles: list[TrackSummary] = Field(default_factory=list)
    pedestrians: list[TrackSummary] = Field(default_factory=list)
    # S3 object keys, not URLs — presigned links expire, so they are minted per read
    # the same way file downloads are.
    #
    # EMPTY, on everything the worker writes, and that is settled rather than pending.
    # Frames are re-derived from the source when somebody opens the detail view, which
    # the row's source_id and frame_index are there to make possible. The field stays
    # for the case that would need it — a durable artifact baked at confirmation time,
    # or a stream, which has no source to seek back into.
    frames: list[str] = Field(default_factory=list)


class ViolationCreate(BaseModel):
    """What the worker has when a rule fires. Everything else is defaulted."""

    site_id: str
    # The source version the job was pinned to, and the frame within it. Together they
    # are what lets evidence be re-derived on demand instead of uploaded here — see the
    # DDL. The worker has both already: the job message carries the source it was
    # created against, and a Violation carries its own frame index.
    source_id: str
    frame_index: int
    type: ViolationType
    detected_at: datetime
    metadata: ViolationMetadata = Field(default_factory=ViolationMetadata)


class ViolationResponse(BaseModel):
    id: str
    site_id: str
    # Carried on the way out too: whoever renders the detail view is the one that has
    # to fetch the video and seek to the moment.
    source_id: str
    frame_index: int
    type: ViolationType
    status: ViolationStatus
    detected_at: datetime
    explanation: str | None = None
    severity: str | None = None
    created_at: datetime
    updated_at: datetime
    # Absent on list reads, which never join the metadata table.
    metadata: ViolationMetadata | None = None
