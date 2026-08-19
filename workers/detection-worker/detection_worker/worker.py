"""Queue consumer entrypoint.

A stub: it takes jobs off the queue and logs them. The pipeline the LLD describes —
source, detect, track, evaluate, store — hangs off `handle` when it exists. What this
proves for now is the hop itself, that a job site-service pushed arrives in another
process intact.
"""

import logging
from typing import Callable

from shared.models.detection import DetectionJob
from shared.queue.client import from_config

logger = logging.getLogger(__name__)


def handle(job: DetectionJob) -> None:
    """Do the work. For now, the job's arrival is the work."""
    logger.info(
        "detection job %s site=%s frames=%d-%d types=%s",
        job.id,
        job.site_id,
        job.frame_range.start,
        job.frame_range.end,
        [t.value for t in job.types],
    )


def run(
    queue,
    handle: Callable[[DetectionJob], None] = handle,
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
    run(from_config())


if __name__ == "__main__":
    main()
