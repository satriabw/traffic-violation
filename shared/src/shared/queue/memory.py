"""An in-process queue with the same shape as RedisQueue.

Not test scaffolding: it is what lets the worker run through a job without Redis
anywhere, and it saves both test suites from each growing their own fake.
"""

from collections import deque

from shared.models.detection import DetectionJob


class InMemoryQueue:
    def __init__(self):
        self._jobs: deque[DetectionJob] = deque()

    def enqueue(self, job: DetectionJob) -> None:
        self._jobs.append(job)

    def consume(self, timeout: int = 0) -> DetectionJob | None:
        """The next job, or None once drained.

        Nothing can arrive while a single-threaded caller is blocked here, so an empty
        queue returns immediately rather than honouring `timeout` — that is what stops
        the worker loop against this queue while the same loop runs forever against
        Redis.
        """
        return self._jobs.popleft() if self._jobs else None
