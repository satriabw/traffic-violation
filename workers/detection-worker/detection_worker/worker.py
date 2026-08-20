"""Queue consumer entrypoint.

It takes jobs off the queue and reads the frames each one asks for. That is the whole
worker today: the rest of the pipeline the LLD describes — detect, track, evaluate,
store — hangs off the frames `make_handler` iterates. Reading lands on its own first
so the hop from a queued job to decoded pixels is proven before a model sits on top
of it.
"""

import logging
from typing import Any, Callable, Iterable

from shared.models.detection import DetectionJob, FrameRange
from shared.queue.client import from_config
from shared.s3.client import presigned_get

from detection_worker.reader import read_frames

logger = logging.getLogger(__name__)


def make_handler(
    sign: Callable[[str], str] = presigned_get,
    read: Callable[[str, FrameRange], Iterable[tuple[int, Any]]] = read_frames,
) -> Callable[[DetectionJob], None]:
    """Build the job handler, with its two collaborators injectable.

    A factory rather than a wider `handle` signature, so `run` keeps taking a plain
    `Callable[[DetectionJob], None]` and tests substitute a fake signer and reader
    without S3 or a video anywhere in reach.
    """

    def handle(job: DetectionJob) -> None:
        # Signed here rather than at enqueue time: a presigned url expires, and one
        # that died in a backlog would fail after the worker had started. The key in
        # the message is immutable, so this is always safe to do late. See JobSource.
        url = sign(job.source.key)

        read_count = 0
        for _index, _frame in read(url, job.frame_range):
            # Where detection will go. Counting is deliberately all this does — it is
            # the smallest thing that proves every requested frame decoded.
            read_count += 1

        logger.info(
            "detection job %s site=%s source=%s v%d frames=%d-%d read=%d types=%s",
            job.id,
            job.site_id,
            job.source.source_id,
            job.source.version,
            job.frame_range.start,
            job.frame_range.end,
            read_count,
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
    logger.info("detection-worker waiting for jobs")
    run(from_config(), make_handler())


if __name__ == "__main__":
    main()
