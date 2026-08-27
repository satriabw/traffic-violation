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


class Severity(str, Enum):
    """How much the violation mattered, not how certain the detector was.

    Three bands rather than a score, because the thing being graded is a judgement a
    reviewer has to agree with: an explainer that says HIGH has to name what made it
    high, and a reviewer disagreeing wants to argue with a band and its reasons rather
    than with the difference between 0.71 and 0.68.

    Severity is about the risk imposed on other people, so identical driving earns
    different bands in different circumstances — an empty junction and a crosswalk in
    use are not the same event. What separates them is only ever in the evidence, which
    is why `severity_basis` below is not optional.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ViolationExplanation(BaseModel):
    """What an explainer returns for one violation.

    Stored whole in `traffic_violations.explanation_json`, with `explanation` and
    `severity` also copied onto their own columns — those two are what the list
    endpoint renders, and a list that had to parse a JSON blob per row to show a
    severity chip would defeat the flat columns entirely.
    """

    # WHETHER THE DETECTION HOLDS UP, asked separately from what it was. A detector
    # that tracked the wrong vehicle, misread a signal region, or fired on a vehicle
    # that entered legally and was still clearing the junction produces a row that
    # looks exactly like a real violation, and an explainer given no way to say "this
    # one does not stand" will write a fluent account of something that did not happen.
    flag_sustained: bool = True
    explanation: str
    severity: Severity
    # WHAT THE BAND RESTS ON, and it has to be things visible in the evidence rather
    # than the reasoning that got there. Separate from `observations` because a
    # reviewer overturning a severity wants the two or three facts that decided it, not
    # everything the explainer happened to notice.
    severity_basis: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    # WHAT THE EXPLAINER DID NOT TRUST, and the field that earns the JSON column. An
    # explainer handed a speed derived through a calibration the violation never had
    # will notice it is impossible; with nowhere to record that, the doubt either
    # disappears from the record or, worse, silently grades the severity. Empty means
    # nothing looked wrong — not that nothing was checked.
    evidence_concerns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ViolationCreate(BaseModel):
    """What the worker has when a rule fires. Everything else is defaulted."""

    site_id: str
    # The source version the job was pinned to, and the frame within it. Together they
    # are what evidence-worker seeks to when it cuts this violation's thumbnail and
    # clip — and the only thing that could ever cut them again. The worker has both
    # already: the job message carries the source it was created against, and a
    # Violation carries its own frame index.
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
    # Carried on the way out too: a reader that wants the footage itself, rather than
    # the cut of it on the row, is the one that has to seek to the moment. None on a
    # violation recorded before these existed — nothing can cut its evidence, which is
    # exactly what a NULL `evidence_status` says beside it.
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
    # The same two objects, signed. Minted per read from the keys above rather than
    # stored beside them, exactly as FileResponse.download_url is minted from
    # files.url — a presigned link expires, so one written to a row would be a fact
    # that stops being true without anything having changed.
    #
    # Both None wherever the key is. A key is only ever non-null on a row the worker
    # finished, because set_evidence rewrites both keys on every transition including
    # the Nones a 'pending' or 'failed' write carries — so there is no state where a
    # key survives without an object behind it, and no second check on
    # `evidence_status` to make here.
    thumbnail_url: str | None = None
    clip_url: str | None = None
    created_at: datetime
    updated_at: datetime
    # Absent on list reads, which never join the metadata table.
    metadata: ViolationMetadata | None = None
    # The rest of the explanation, parsed back out of `explanation_json`. `explanation`
    # and `severity` above are the same answer's two flat fields and stay populated
    # either way, so a list read losing this loses detail and never the verdict.
    #
    # None on a violation nothing has explained, and also on every list read — the same
    # two reasons `metadata` is None, and distinguishable the same way: `status` says
    # whether an explanation exists.
    explanation_detail: ViolationExplanation | None = None


class ViolationListResponse(BaseModel):
    """A page of violations. Same shape as SiteListResponse, deliberately.

    `items` never carries `metadata`: the blob is a separate table precisely so a page
    of violations does not drag every track's trajectory along with it, and a list that
    joined it would undo that at exactly the scale it was split to survive.
    """

    items: list[ViolationResponse]
    # Every violation matching the filter, not the length of `items` — a caller
    # paginating needs to know what it is paginating through.
    total: int
    limit: int
    offset: int
