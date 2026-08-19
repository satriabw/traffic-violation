import json

from shared.models.detection import DetectionJob, FrameRange, ViolationType
from shared.queue.client import RedisQueue
from shared.queue.memory import InMemoryQueue

QUEUE_NAME = "detection:jobs"


def _job(job_id: str = "job-1", start: int = 0, end: int = 100) -> DetectionJob:
    return DetectionJob(
        id=job_id,
        site_id="site-1",
        frame_range=FrameRange(start=start, end=end),
        types=[ViolationType.RED_LIGHT_RUNNING],
    )


class FakeRedis:
    """Enough of redis-py for the two commands the queue uses.

    Injected rather than patched, so the tests exercise the real serialisation and
    the real command wiring without a server anywhere near them.
    """

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.brpop_calls: list[tuple] = []

    def lpush(self, name: str, value: str) -> int:
        self.lists.setdefault(name, []).insert(0, value)
        return len(self.lists[name])

    def brpop(self, name: str, timeout: int = 0):
        self.brpop_calls.append((name, timeout))
        values = self.lists.get(name, [])
        if not values:
            # What a real BRPOP returns when it times out. With timeout=0 it would
            # block instead, which no test can wait on.
            return None
        return (name, values.pop())


# --- InMemoryQueue --------------------------------------------------------


def test_in_memory_round_trips_a_job_unchanged():
    queue = InMemoryQueue()
    job = _job()

    queue.enqueue(job)

    assert queue.consume() == job


def test_in_memory_is_fifo():
    queue = InMemoryQueue()
    for i in range(3):
        queue.enqueue(_job(job_id=f"job-{i}"))

    assert [queue.consume().id for _ in range(3)] == ["job-0", "job-1", "job-2"]


def test_in_memory_consume_returns_none_when_drained():
    # This is what lets the worker loop terminate against an in-memory queue while
    # blocking forever against Redis — one loop, two behaviours.
    assert InMemoryQueue().consume() is None


# --- RedisQueue -----------------------------------------------------------


def test_redis_enqueue_pushes_json_onto_the_named_list():
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME)

    queue.enqueue(_job())

    assert json.loads(redis.lists[QUEUE_NAME][0]) == {
        "id": "job-1",
        "site_id": "site-1",
        "frame_range": {"start": 0, "end": 100},
        "types": ["red_light_running"],
    }


def test_redis_consume_parses_a_brpop_reply():
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME)
    job = _job()
    queue.enqueue(job)

    assert queue.consume() == job


def test_redis_queue_is_fifo():
    # LPUSH pushes to the head and BRPOP pops the tail; getting that pair backwards
    # would silently make the queue a stack.
    queue = RedisQueue(FakeRedis(), QUEUE_NAME)
    for i in range(3):
        queue.enqueue(_job(job_id=f"job-{i}"))

    assert [queue.consume().id for _ in range(3)] == ["job-0", "job-1", "job-2"]


def test_redis_consume_returns_none_on_timeout():
    assert RedisQueue(FakeRedis(), QUEUE_NAME).consume() is None


def test_redis_consume_blocks_indefinitely_by_default():
    redis = FakeRedis()

    RedisQueue(redis, QUEUE_NAME).consume()

    # timeout=0 is BRPOP for "wait forever". A worker that polled instead would burn
    # a round trip per second doing nothing.
    assert redis.brpop_calls == [(QUEUE_NAME, 0)]
