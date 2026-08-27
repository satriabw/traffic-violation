"""What happens when a rule fires.

Two things: the `Violation` and the windows around it become a `ViolationCreate`, and
once that is written a cut of the footage is asked for.

The SQL that writes one lives in `shared.db.violations`, because evidence-worker writes
to the same tables afterwards and one copy of those statements is the only way they
stay in step with the schema. `record` is re-exported here so callers keep one import
for the whole job of recording a violation, which is how `worker.py` already reads.
"""

import logging
from datetime import datetime, timedelta

from evidence_collector import TrackWindow
from shared.db.violations import record as record  # re-exported; see the module docstring
from shared.db.violations import set_evidence
from shared.models.detection import DetectionJob, ViolationType
from shared.models.evidence import EvidenceJob
from shared.models.violation import (
    EvidenceStatus,
    TrackSummary,
    ViolationCreate,
    ViolationMetadata,
)
from violation_detector import PEDESTRIANS, Violation

from detection_worker import context

logger = logging.getLogger(__name__)


def queue_evidence(con, queue, violation_id: str, evidence_seconds: float) -> None:
    """Ask for this violation's thumbnail and clip to be cut.

    `evidence_seconds` is the site's, resolved once when the job's context was loaded
    and already used to size the ring buffer this violation's window came out of. It
    travels on the message because the violation row cannot answer for it — it lives in
    a configuration document in object storage — and because passing the number the
    record was kept over is what makes the clip cover the same frames the record
    describes. Reading it again over there would be an S3 fetch per violation for an
    answer this process is holding.

    PENDING IS WRITTEN FIRST, AND THE ORDER IS THE WHOLE OF THE DESIGN HERE. Enqueue
    first and evidence-worker — a different process, possibly already idle — can finish
    the cut and write 'ready' before this line runs, and 'pending' would then overwrite
    a violation that already has its evidence. Writing first cannot lose that race: the
    only thing that can follow it is the worker's own verdict.

    A QUEUE THAT WILL NOT TAKE THE JOB DOES NOT STOP THE DETECTOR. This is the one place
    in the worker that swallows an exception, and it is deliberate: everything else here
    raises because losing a job silently is worse than stopping loudly, but the job in
    question has already been recorded — the violation is safe on disk, and giving up
    GPU throughput because a second queue is unreachable trades the expensive half of
    the pipeline for the cheap one.

    So the row is marked failed instead, which is a true statement — nothing is going to
    produce this evidence — and one a backfill can find. `logger.exception` rather than
    a message, because a broad `except` that hides its traceback is how a bug in here
    would look exactly like Redis being down.
    """
    set_evidence(con, violation_id, EvidenceStatus.PENDING)
    try:
        queue.enqueue(
            EvidenceJob(violation_id=violation_id, evidence_seconds=evidence_seconds)
        )
    except Exception:
        logger.exception("could not queue evidence for violation %s", violation_id)
        set_evidence(con, violation_id, EvidenceStatus.FAILED)


def summary(window: TrackWindow) -> TrackSummary:
    """One track's window, in the shape the metadata blob keeps.

    A rename and nothing else. The four parallel lists are the same four the window
    already holds, in the same order — the evidence package answers in its own
    vocabulary because it depends on nothing, and this is the whole cost of that.

    Nothing is filled in on the way past. A frame nothing projected keeps its None,
    which is why `TrackSummary` accepts one: a job with no calibration writes a
    trajectory of Nones beside a full set of boxes, and that is an honest record of
    what was known rather than a plausible one of what was not.
    """
    return TrackSummary(
        track_id=window.track_id,
        trajectory=list(window.positions),
        speed=list(window.speeds),
        frame_idxs=list(window.frame_indices),
        bboxes=list(window.bboxes),
    )


def detected_at(anchor: datetime, frame_index: int, fps: float | None) -> datetime:
    """When in the footage this happened, as a moment.

    The frame index is the offset; `anchor` turns it into a time. A source with no
    probed frame rate cannot convert one into the other at all, so it reports the
    anchor itself rather than inventing a rate — every violation in such a job lands on
    the same timestamp, which is visibly wrong rather than quietly wrong.
    """
    if not fps or fps <= 0:
        return anchor
    return anchor + timedelta(seconds=frame_index / fps)


def scene(
    windows: dict[int, TrackWindow], violator_track_id: int | None = None
) -> ViolationMetadata:
    """Every track the buffer held, split by what the detector called each one.

    THE SPLIT IS BY CLASS, NOT BY INVOLVEMENT. `class_names` is already on the window,
    recorded per frame and resolved by the evidence package's own most-common rule, so
    this reads a label rather than deciding anything. Which of these was in the crossing
    is a question about polygons; the row pins `configuration_id` so a reader can answer
    it, and nothing here computes geometry to guess.

    Anything the rules have no vocabulary for — a tracked traffic light, a class the
    model had no name for — lands in `vehicles`. Wrong bucket, kept record: dropping it
    would contradict recording the scene, and no rule that ships can produce one.

    Ordered by track id, because `window_for` already answers that way and a blob that
    came out differently on two identical runs would be a difference nobody meant.
    """
    vehicles, pedestrians = [], []
    for track_id in sorted(windows):
        window = windows[track_id]
        bucket = pedestrians if window.class_name in PEDESTRIANS else vehicles
        bucket.append(summary(window))
    return ViolationMetadata(
        vehicles=vehicles, pedestrians=pedestrians, violator_track_id=violator_track_id
    )


def to_create(
    job: DetectionJob,
    violation: Violation,
    windows: dict[int, TrackWindow],
    job_context: context.JobContext,
) -> ViolationCreate:
    """A firing rule and its record, as the row that gets written.

    `violation.frame_index`, never the index of the frame being analysed when this was
    produced. A module working on a clip reports several frames late, and recording the
    loop's position would misdate it by the length of the window.

    The whole context rather than the anchor alone. Three of the values written here —
    the anchor, and the two documents this was judged against — are facts about the
    job's context that always travel together and always come from the same object, so
    passing that object is one parameter instead of three. It also means the pending fix
    to `detected_at` (footage time rather than upload time) lands inside `context` with
    no signature to change out here.

    THE WHOLE SCENE IS SUMMARISED, and `violator_track_id` is what keeps the accusation
    legible inside it. `pedestrians` is filled for the first time here — for the
    pedestrian rule, whose whole subject the module never names in what it returns, and
    for the red-light rule too whenever somebody happened to be there. Empty stays the
    ordinary answer on an empty crossing.

    `frames` stays empty by design. The thumbnail and the clip are columns on the row,
    cut by evidence-worker once this has been written — not entries in this list.
    """
    return ViolationCreate(
        site_id=job.site_id,
        source_id=job.source.source_id,
        frame_index=violation.frame_index,
        calibration_id=job_context.calibration_id,
        configuration_id=job_context.configuration_id,
        type=ViolationType(violation.type),
        detected_at=detected_at(
            job_context.source_created_at, violation.frame_index, job.source.fps
        ),
        metadata=scene(windows, violator_track_id=violation.track_id),
    )
