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


class EvidenceStrength(str, Enum):
    """How much of the case the record itself settles, as against the detector's word.

    Not the same question as `Severity`, and the two are graded independently: a clerk
    can be looking at a serious event the record barely supports, or a trivial one it
    establishes completely. Conflating them is the mistake this enum exists to make
    difficult.

    STRONG IS CURRENTLY UNREACHABLE, and that is a fact about the pipeline rather than
    about any case. The detector computes the things that actually convict — which
    signal governed the lane, when it turned red, which crossing somebody was standing
    in — and keeps none of them; see `violation_detector.context` beside
    `Violation`, which carries a type, a track id and a frame. Until that changes, no
    explanation can confirm the element the offence turns on, and the honest ceiling is
    MEDIUM. The band is defined anyway because the fix is a schema change rather than a
    redesign, and a reviewer reading MEDIUM everywhere deserves to know it is the system
    talking and not their case.
    """

    # The record establishes the offence on its own terms.
    STRONG = "STRONG"
    # Consistent with the flag, with something load-bearing taken on trust.
    MEDIUM = "MEDIUM"
    # The record cannot get past the detector's assertion; the footage has to settle it.
    WEAK = "WEAK"


class PlateRecoverability(str, Enum):
    """What could still be done about the plate — never what it says.

    There is no plate anywhere in this system: no ANPR, no OCR, no column, nothing that
    ever reads one. So this is a routing judgement for whoever picks the violation up,
    and the three values are three different next actions rather than three degrees of
    legibility.

    A ten-arm study put this under deliberate pressure — "the case is held up", "partial
    reads are useful" — and got thirteen refusals out of thirteen. The guard is cheap and
    the thing it guards against is naming an innocent registered owner off a fabricated
    read, so it stays.
    """

    # Worth putting through recognition, if recognition ever exists here.
    ANPR_RERUN = "anpr_rerun"
    # Worth pulling the footage and reading by eye.
    MANUAL_READ = "manual_read"
    # Nothing in the record settles it either way.
    INCONCLUSIVE = "inconclusive"


class LicensePlateAssessment(BaseModel):
    """Whether the plate is worth chasing, and why."""

    recoverability: PlateRecoverability
    # WRITTEN FOR THE PERSON REVIEWING THE CASE, which rules out the measurements it
    # would be natural to quote. "Small and distant in frame" is the finding; the pixel
    # width behind it is how the finding was reached and belongs nowhere near a clerk.
    reasoning: str


class ViolationExplanation(BaseModel):
    """A note to the clerk who has to decide what happens to one violation.

    NOT A REPORT ABOUT THE RECORD, which is what it used to be and why it read like a
    machine describing its own input. The reader is a person who will approve the
    violation, reject it, hold it, or send it back for reprocessing, and who is looking
    at footage while they do. Everything here is written for them.

    THE CONTRACT ON EVERY STRING FIELD BELOW: no track ids, no frame indices, no pixel
    measurements. A clerk does not know what a track id is and should not have to; the
    row already carries `id`, `source_id` and `frame_index` for the supervisor auditing
    the decision afterwards, so nothing is lost by keeping them out of the prose. The
    enforcement is upstream rather than here — the prompt is not given the identifiers
    in the first place, because a vocabulary the model never sees is one it cannot use.

    Times, not frames. The source's frame rate turns one into the other, and seconds are
    what somebody scrubbing a clip actually needs.

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
    #
    # ONLY AN ACTIVE CONTRADICTION FLIPS THIS. Something the record merely fails to
    # settle lowers `evidence_strength` and says so; it is not a rejection, because
    # rejecting is the clerk's to do and "we cannot tell from here" is not the same
    # statement as "this did not happen".
    flag_sustained: bool = True
    explanation: str
    severity: Severity
    # WHAT THE BAND RESTS ON, and it has to be things visible in the evidence rather
    # than the reasoning that got there. Separate from `observations` because a
    # reviewer overturning a severity wants the two or three facts that decided it, not
    # everything the explainer happened to notice.
    severity_basis: list[str] = Field(default_factory=list)
    # HOW MUCH OF THIS THE RECORD ITSELF SETTLES, which is the first thing a clerk wants
    # and the thing the old shape had nowhere to put.
    #
    # None means an explanation written before this field existed, never a grade the
    # explainer declined to give — the same thing None means on `calibration_id` and
    # `evidence_status`. It has to default, because `explanation_json` is parsed straight
    # back into this model and rows explained under the old shape are still read.
    evidence_strength: EvidenceStrength | None = None
    # The two or three things carrying that band, in the clerk's terms: what the footage
    # would have to settle, what the record already establishes.
    evidence_basis: list[str] = Field(default_factory=list)
    # None on an explanation written before this existed, and on one where nothing about
    # the plate could be assessed at all.
    license_plate: LicensePlateAssessment | None = None
    # ANYTHING ELSE WORTH THE CLERK'S ATTENTION, and the bar is high on purpose: a line
    # earns its place only if a clerk could not get it by reading the file themselves.
    # A count is not a finding. Two facts that mean something once connected — objects
    # counted as traffic that never move, a person detected for a twentieth of a second
    # — are exactly what this is for.
    observations: list[str] = Field(default_factory=list)
    # WHAT NOT TO TRUST, phrased as what to do about it. An explainer handed speeds
    # derived through a calibration that is producing hundreds of km/h will notice they
    # are impossible; with nowhere to record that, the doubt either disappears from the
    # record or, worse, silently grades the severity. Empty means nothing looked wrong —
    # not that nothing was checked.
    #
    # The mechanism stays out. "Ignore any speed on this record, the camera's distance
    # calibration is faulty" is the useful sentence; how many tracks failed which
    # threshold is how that was worked out, and is not the clerk's problem.
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
