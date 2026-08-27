"""An in-process queue with the same shape as RedisQueue.

Not test scaffolding: it is what lets a worker run through a job without Redis
anywhere, and it saves every test suite from growing its own fake.

Carries whatever it is given, unlike RedisQueue. Nothing is serialised here, so there
is no message type to parse back into and nothing for the caller to declare — the
generic is on the class for readers and type checkers, not for the runtime.
"""

from collections import deque
from typing import Generic, TypeVar

Message = TypeVar("Message")


class InMemoryQueue(Generic[Message]):
    def __init__(self):
        self._jobs: deque[Message] = deque()

    def enqueue(self, job: Message) -> None:
        self._jobs.append(job)

    def consume(self) -> Message | None:
        """The next job, or None once drained.

        Nothing can arrive while a single-threaded caller is blocked here, so an empty
        queue returns immediately. That None is what stops the worker loop against this
        queue, while RedisQueue.consume simply never returns one.
        """
        return self._jobs.popleft() if self._jobs else None
