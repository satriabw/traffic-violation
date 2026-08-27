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


class EvidenceStatus(str, Enum):
    """How far evidence-worker has got with one violation's thumbnail and clip.

    Deliberately not folded into ViolationStatus. That one tracks the LLM explanation,
    this one tracks a cut of the footage, and nothing sequences them — a violation can
    be explained while its clip is still cutting. One enum over both would have to
    enumerate the product of the two.

    There is no member for "nothing will ever build this". That is NULL on the row, and
    it means the violation predates evidence-worker: no keys, and no queued job to
    produce them. `None` here says so, which is why every field below is optional.
    """

    # Queued, not started. Set by whoever enqueues the job, so a row is never briefly
    # indistinguishable from one nobody asked to build.
    PENDING = "pending"
    # Both objects are in storage and the keys on the row resolve.
    READY = "ready"
    # The worker gave up. Terminal — nothing retries yet — so a reader should offer the
    # source and frame index instead of waiting.
    FAILED = "failed"


class TrackSummary(BaseModel):
    """One object's path through the window that produced the violation.

    The four lists are parallel — index i of each describes the same frame — which is
    what makes a trajectory point matchable to the box it came from. `track_id` is the
    tracker's id, and is only meaningful within the job that assigned it.
    """

    track_id: int
    # Ground-plane metres, and None on any frame nothing projected — a job with no
    # calibration has no ground plane, so every entry is None and `bboxes` is the whole
    # of what was seen. Nothing substitutes image coordinates here: a position that is
    # sometimes metres and sometimes pixels is a hundred-fold error nobody downstream
    # could detect, and the box's own bottom edge is one line for a reader that wants
    # somewhere to draw.
    trajectory: list[tuple[float, float] | None] = Field(default_factory=list)
    # None wherever nothing measured one — during a filter's warmup, and throughout a
    # window that was never projected. Not 0.0, which is a speed somebody could have
    # measured.
    speed: list[float | None] = Field(default_factory=list)
    frame_idxs: list[int] = Field(default_factory=list)
    # (x1, y1, x2, y2), the same convention supervision uses.
    bboxes: list[tuple[float, float, float, float]] = Field(default_factory=list)


class ViolationMetadata(BaseModel):
    """The json_blob, typed. Shape is the LLD's."""

    # WHO WAS ON THE SCENE, not who was to blame. Every track the evidence buffer held
    # when the rule fired, split by what the detector called each one — the vehicle
    # convicted, the pedestrian it drove at, and the car queued behind that had nothing
    # to do with either. Deciding which of them mattered means testing their boxes
    # against the regions in force, and the row pins `configuration_id` so a reader can
    # do exactly that; the worker records what was there and interprets none of it.
    vehicles: list[TrackSummary] = Field(default_factory=list)
    pedestrians: list[TrackSummary] = Field(default_factory=list)
    # WHICH TRACK WAS CONVICTED. Load-bearing, and only since the lists above became
    # the whole scene: a reader used to be able to take `vehicles[0]`, because there was
    # never more than one. Now nothing else in here distinguishes the driver who failed
    # to yield from the driver waiting behind them.
    #
    # The tracker's id, so it means something only alongside the run that assigned it —
    # which is the same scope as the windows it indexes into, and why it lives in the
    # blob beside them rather than on the row. None on a violation whose rule reported
    # no track, and on everything written before this existed.
    violator_track_id: int | None = None
    # S3 object keys, not URLs — presigned links expire, so they are minted per read
    # the same way file downloads are.
    #
    # STILL EMPTY, on everything anything writes. The durable artifact this was held
    # open for now exists, and it is not here: `thumbnail_key` and `clip_key` are
    # columns on the violation row, because the list endpoint never joins this table
    # and a thumbnail it cannot reach is a thumbnail it cannot render.
    #
    # What is left for this field is the case that is genuinely per-frame and genuinely
    # a set — an explainer picking out the frames it wants to reason over, rather than
    # the two fixed artifacts a reviewer opens. Nothing writes it yet.
    frames: list[str] = Field(default_factory=list)


class ViolationCreate(BaseModel):
    """What the worker has when a rule fires. Everything else is defaulted."""

    site_id: str
    # The source version the job was pinned to, and the frame within it. Together they
    # are what lets evidence be re-derived on demand instead of uploaded here — see the
    # DDL. The worker has both already: the job message carries the source it was
    # created against, and a Violation carries its own frame index.
    #
    # REQUIRED HERE, though the columns are nullable. The columns had to give way to
    # rows that predate them; nothing being written now has that excuse, and this is
    # where the guarantee lives instead.
    source_id: str
    frame_index: int
    # What this was judged against, pinned the same way the source is. The job message
    # already carries both, resolved while "active" still meant what the caller asked
    # for, so the worker has them without a lookup of its own.
    #
    # OPTIONAL HERE, where source_id and frame_index are required, and the difference is
    # real rather than an oversight. Every job has a source; not every job has a
    # calibration, because a site with a video and no camera model is an ordinary site
    # and detection runs anyway. None says the site had none — it is the same absence
    # DetectionJob.calibration_version already carries, and there is no guarantee to
    # make here that would be true.
    calibration_id: str | None = None
    configuration_id: str | None = None
    type: ViolationType
    detected_at: datetime
    metadata: ViolationMetadata = Field(default_factory=ViolationMetadata)


class ViolationResponse(BaseModel):
    id: str
    site_id: str
    # Carried on the way out too: whoever renders the detail view is the one that has
    # to fetch the video and seek to the moment. None on a violation recorded before
    # these existed — its evidence cannot be re-derived, and a reader that pretended
    # otherwise would send someone looking for a video nothing named.
    source_id: str | None = None
    frame_index: int | None = None
    # Carried out too, because the reader is the one that needs them: filtering a site's
    # violations down to the setup it runs under now is a comparison against these, and
    # drawing the evidence means resolving the polygons this was actually judged with
    # rather than whatever is current. None either because the site had none, or because
    # the violation predates the columns — the two are indistinguishable here, which is
    # why a filter has to say which it means.
    calibration_id: str | None = None
    configuration_id: str | None = None
    type: ViolationType
    status: ViolationStatus
    detected_at: datetime
    explanation: str | None = None
    severity: str | None = None
    # Carried on list reads as well as detail ones, which is the point of them being
    # columns: rendering a page of violations means a thumbnail each, and a reader that
    # had to open every violation to find its key would defeat the pre-baking entirely.
    #
    # All three None together on a violation recorded before evidence-worker existed.
    # A reader has to tell that apart from PENDING — see EvidenceStatus — because the
    # first will never become the second.
    thumbnail_key: str | None = None
    clip_key: str | None = None
    evidence_status: EvidenceStatus | None = None
    created_at: datetime
    updated_at: datetime
    # Absent on list reads, which never join the metadata table.
    metadata: ViolationMetadata | None = None
