"""Queue consumer entrypoint.

It takes jobs off the queue, resolves the calibration and configuration each one was
created against, reads the frames it asks for, and hands each frame to an analyzer.
The pipeline now runs end to end: detection, tracking, ground positions, and the rules
a site's configuration asks for.

A firing rule now becomes a row. `detection_worker.violations` turns the `Violation`
and the window that came with it into a `ViolationCreate` and writes it; the pipeline
runs end to end.

NO FRAMES ARE UPLOADED HERE, and that is settled rather than pending. The record keeps
the frame index, and the source is an immutable object in storage that can be seeked
back into, so the pixels are re-derived when somebody opens the detail view — by
whatever knows how to draw them then, rather than by this process guessing now.

Everything here is per-job. Per-frame work lives in `detection_worker.analysis`.
"""

import logging
from typing import Any, Callable, Iterable

from shared.models.detection import DetectionJob, FrameRange
from shared.models.violation import ViolationCreate
from shared.queue.client import from_config
from shared.s3.client import presigned_get

from detection_worker import context, violations
from detection_worker.analysis.frame_analyzer import FrameAnalyzer, make_analyzer
from detection_worker.db import get_db
from detection_worker.detection.model import from_config as model_from_config
from detection_worker.video.reader import read_frames

logger = logging.getLogger(__name__)


def make_handler(
    load_context: Callable[[DetectionJob], context.JobContext],
    new_analyzer: Callable[[DetectionJob, context.JobContext], FrameAnalyzer],
    save: Callable[[ViolationCreate], str],
    sign: Callable[[str], str] = presigned_get,
    read: Callable[[str, FrameRange], Iterable[tuple[int, Any]]] = read_frames,
) -> Callable[[DetectionJob], None]:
    """Build the job handler, with its collaborators injectable.

    A factory rather than a wider `handle` signature, so `run` keeps taking a plain
    `Callable[[DetectionJob], None]` and tests substitute a fake signer, reader and
    analyzer without S3, a video or a weights file anywhere in reach.

    Neither argument has a default, unlike the two below them. Each of those has a
    sensible production value; these two do not. `load_context` needs a database
    connection, and defaulting it would mean quietly opening one — turning a missing
    TRAFFIC_DB_PATH into a job that runs with no calibration rather than a worker that
    will not start. `new_analyzer` needs the process's detection model, which is loaded
    once in `main` and has nowhere to hide.
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

        # Once per job, before the loop. Inside it, every frame would land in an
        # analyzer whose tracker had never seen the previous one, and nothing would
        # ever hold an id for longer than a single frame. Built after the context
        # because it is built *from* it — a job's calibration is what the trajectory
        # collector projects with.
        analyzer = new_analyzer(job, job_context)
        # Read once, before the loop. It is a property of the job, and asking the
        # analyzer per frame would be the same answer every time.
        capacity = analyzer.evidence_capacity

        read_count = 0
        detection_count = 0
        violation_count = 0
        evidence_count = 0
        recorded_count = 0
        # Windows that came back shorter than the ring can hold. A track seen for half
        # a second only has half a second of history and is unremarkable, but so is a
        # violation early in a chunk that reaches back past its own first frame — and
        # the second one is a truncated record that looks exactly like a short one.
        # Counted rather than warned about per violation, because the number that
        # matters is whether it is a handful or all of them.
        short_windows = 0
        seen_tracks: set[int] = set()
        located_tracks: set[int] = set()

        for index, frame in read(url, job.frame_range):
            result = analyzer.analyze(frame, index)

            ids = result.track_ids
            read_count += 1
            detection_count += len(result.detections)
            seen_tracks.update(ids)
            located_tracks.update(result.trajectories)
            violation_count += len(result.violations)
            evidence_count += len(result.evidence)
            for violation in result.violations:
                # Written as they are found rather than collected and written at the
                # end. A job that dies half way through has recorded what it saw up to
                # then, which is worth more than nothing — and each write is its own
                # transaction, so there is no batch to lose.
                save(
                    violations.to_create(
                        job,
                        violation,
                        result.evidence.get(violation.track_id),
                        job_context,
                    )
                )
                recorded_count += 1
            short_windows += sum(
                1 for window in result.evidence.values() if len(window) < capacity
            )

            # Per-frame detail is DEBUG because a 30-second chunk is ~900 of these,
            # and the summary below is what anyone watching a normal run wants.
            logger.debug(
                "job %s frame %d detections=%d ids=%s",
                job.id,
                result.index,
                len(result.detections),
                ids,
            )

        # Once, after the loop. A rule reports on the frame it was given and holds
        # nothing, but a module working on a clip is still holding a partial one here,
        # and without this drain the last seconds of every chunk would be dropped in
        # silence. These carry no window — see FrameAnalyzer.finish — and are recorded
        # anyway: a violation with no history is still a violation.
        for violation in analyzer.finish():
            save(violations.to_create(job, violation, None, job_context))
            violation_count += 1
            recorded_count += 1

        logger.info(
            "detection job %s site=%s source=%s v%d calib=%s config=%s frames=%d-%d "
            "read=%d detections=%d tracks=%d located=%d violations=%d evidence=%d "
            "short=%d recorded=%d types=%s",
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
            # How many of those were put on the ground. Zero where the site has no
            # calibration, and short of `tracks` where some object never had a box
            # whose bottom edge met the ground — which is the difference between "we
            # saw it" and "we know where it was".
            len(located_tracks),
            # What the rules reported, including anything drained above. Zero for a
            # site with no configuration, and zero for a job that asked for no types —
            # both normal, and neither distinguishable from a clean stretch of footage
            # by this number alone.
            violation_count,
            # How many of those came with a history. Short of `violations` only where
            # two rules convicted one vehicle on one frame, since they share the one
            # window that describes them both.
            evidence_count,
            # How many of those histories were cut short. A few is ordinary — objects
            # that had only just appeared. Most of them, on a job that is not the first
            # chunk of its video, means the window is longer than the overlap between
            # chunks and every record is missing its approach.
            short_windows,
            # Rows written. Equal to `violations` on a run that finished, and the pair
            # is worth logging together anyway: a handler that raised part way through
            # keeps whatever it had already committed, so the two diverging is the
            # signal that a job did not get to the end.
            recorded_count,
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
    # makes it once-per-process: every analyzer the loop below builds reuses this
    # session.
    model = model_from_config()
    # Same reasoning as the model: opened before the queue is touched, so a database
    # that is missing or has no schema stops the worker while it still has no claim on
    # any job.
    con = get_db()
    logger.info("detection-worker waiting for jobs")
    run(
        from_config(),
        make_handler(
            lambda job: context.load(con, job),
            save=lambda violation: violations.record(con, violation),
            new_analyzer=lambda job, job_context: make_analyzer(
                model,
                job.source.fps,
                job_context.calibration,
                job_context.configuration,
                # Canonical values, so the worker holds no table mapping its own
                # ViolationType to the names a configuration document uses. The
                # registry is the only place that knows both.
                [t.value for t in job.types],
                # A number, not a document. Which number is the site's decision, taken
                # where its configuration was resolved.
                seconds=job_context.evidence_seconds,
            ),
        ),
    )


if __name__ == "__main__":
    main()
