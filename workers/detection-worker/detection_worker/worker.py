"""Queue consumer entrypoint.

It takes jobs off the queue, resolves the calibration and configuration each one was
created against, reads the frames it asks for, and runs detection and tracking over
them. What is still missing is the middle of the pipeline: the rule engine, and the
evidence frames a firing rule uploads. Both ends of it exist — tracked detections
come out of `make_handler`, and `detection_worker.violations.record` writes what a
rule decides — so what lands next is the part that connects them.
"""

import logging
from typing import Any, Callable, Iterable

import supervision as sv

from shared.models.detection import DetectionJob, FrameRange
from shared.queue.client import from_config
from shared.s3.client import presigned_get

from detection_worker import context
from detection_worker.db import get_db
from detection_worker.model import DetectionModel
from detection_worker.model import from_config as model_from_config
from detection_worker.reader import read_frames
from detection_worker.tracker import Tracker, make_tracker

logger = logging.getLogger(__name__)


def _track_ids(detections: sv.Detections) -> list[int]:
    """The tracker ids on a frame, or none at all.

    `tracker_id` is None rather than an empty array on a frame the tracker had
    nothing to assign, which is most frames of most footage.
    """
    if detections.tracker_id is None:
        return []
    return [int(tracker_id) for tracker_id in detections.tracker_id]


def make_handler(
    model: DetectionModel,
    load_context: Callable[[DetectionJob], context.JobContext],
    sign: Callable[[str], str] = presigned_get,
    read: Callable[[str, FrameRange], Iterable[tuple[int, Any]]] = read_frames,
    new_tracker: Callable[[float | None], Tracker] = make_tracker,
) -> Callable[[DetectionJob], None]:
    """Build the job handler, with its collaborators injectable.

    A factory rather than a wider `handle` signature, so `run` keeps taking a plain
    `Callable[[DetectionJob], None]` and tests substitute a fake signer, reader,
    model and tracker without S3, a video or a weights file anywhere in reach.

    `model` is passed in rather than built here because building it is expensive and
    it holds no per-job state — one session serves the whole process. `new_tracker`
    is a factory for the opposite reason: a tracker is cheap and holds nothing but
    per-job state, so each job gets its own.

    `load_context` has no default, unlike the collaborators below it. Every one of
    those has a sensible production value; this one needs a database connection, and
    defaulting it would mean quietly opening one — turning a missing TRAFFIC_DB_PATH
    into a job that runs with no calibration rather than a worker that will not start.
    """

    def handle(job: DetectionJob) -> None:
        # Before any decoding: resolving context is cheap and fails fast, and a job
        # naming a calibration that is not there should not first spend minutes
        # reading frames. Pinned to the versions in the message, never to whatever is
        # active now — see detection_worker.context.
        job_context = load_context(job)

        # Signed here rather than at enqueue time: a presigned url expires, and one
        # that died in a backlog would fail after the worker had started. The key in
        # the message is immutable, so this is always safe to do late. See JobSource.
        url = sign(job.source.key)

        # Once per job, before the loop. Inside it, every frame would land in a
        # tracker that had never seen the previous one, and nothing would ever hold
        # an id for longer than a single frame.
        tracker = new_tracker(job.source.fps)

        read_count = 0
        detection_count = 0
        seen_tracks: set[int] = set()

        for index, frame in read(url, job.frame_range):
            detections = model.predict(frame)
            # Every frame, empty or not: the tracker counts frames by counting
            # updates, and skipping one ages every lost track wrongly.
            tracked = tracker.update(detections)

            ids = _track_ids(tracked)
            read_count += 1
            detection_count += len(tracked)
            seen_tracks.update(ids)

            # Per-frame detail is DEBUG because a 30-second chunk is ~900 of these,
            # and the summary below is what anyone watching a normal run wants.
            logger.debug(
                "job %s frame %d detections=%d ids=%s", job.id, index, len(tracked), ids
            )

        logger.info(
            "detection job %s site=%s source=%s v%d calib=%s config=%s frames=%d-%d "
            "read=%d detections=%d tracks=%d types=%s",
            job.id,
            job.site_id,
            job.source.source_id,
            job.source.version,
            # Logged as the versions rather than the documents: this is the line
            # someone reads when a run looks wrong, and "which calibration was this?"
            # is the question. "-" where the site had none.
            job.calibration_version if job_context.calibration else "-",
            job.configuration_version if job_context.configuration else "-",
            job.frame_range.start,
            job.frame_range.end,
            read_count,
            detection_count,
            # Distinct ids, so this is roughly "how many objects did we see", against
            # detection_count's "how many boxes". Ids restart at 1 for every job, so
            # this number is only ever meaningful within one chunk.
            len(seen_tracks),
            [t.value for t in job.types],
        )

    return handle


def run(
    queue,
    handle: Callable[[DetectionJob], None],
    max_jobs: int | None = None,
) -> int:
    """Consume until the queue is exhausted, or max_jobs have been handled.

    Whether "exhausted" ever happens is the queue's decision, not this loop's: an
    InMemoryQueue returns None once drained, while RedisQueue.consume blocks on BRPOP
    and so keeps a deployed worker running indefinitely. One loop covers both.

    max_jobs exists so tests can bound the loop without signals; nothing else uses it.

    A handler that raises stops the worker rather than dropping the job. Retries and a
    dead-letter queue are marked FUTURE in the LLD, and until they exist, failing
    loudly beats losing work quietly.
    """
    handled = 0
    while max_jobs is None or handled < max_jobs:
        job = queue.consume()
        if job is None:
            break
        handle(job)
        handled += 1
    return handled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Before the queue is touched, so a missing or unreadable model file stops the
    # worker while it still has no claim on any job. Loading it here is also what
    # makes it once-per-process: every job the loop below handles reuses this session.
    model = model_from_config()
    # Same reasoning as the model: opened before the queue is touched, so a database
    # that is missing or has no schema stops the worker while it still has no claim on
    # any job.
    con = get_db()
    logger.info("detection-worker waiting for jobs")
    run(from_config(), make_handler(model, lambda job: context.load(con, job)))


if __name__ == "__main__":
    main()
