import json

from shared.models.detection import DetectionJob, FrameRange, ViolationType
from shared.queue.client import BLOCK_SECONDS, RedisQueue, from_config
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
        # Called at the start of each brpop, so a test can let a job land part way
        # through a wait rather than only before it.
        self.on_brpop = lambda: None

    def lpush(self, name: str, value: str) -> int:
        self.lists.setdefault(name, []).insert(0, value)
        return len(self.lists[name])

    def brpop(self, name: str, timeout: int = 0):
        self.on_brpop()
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


def test_redis_consume_waits_through_empty_windows():
    # An idle worker must survive a quiet queue. consume() keeps waiting rather than
    # returning None, so run()'s "None means drained" only ever fires in memory.
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME)
    job = _job()
    redis.on_brpop = lambda: queue.enqueue(job) if len(redis.brpop_calls) == 2 else None

    assert queue.consume() == job
    assert len(redis.brpop_calls) == 3


def test_redis_blocks_in_bounded_windows():
    # The regression this file exists for. BRPOP with timeout=0 blocks forever, which
    # outlives redis-py's 5s socket timeout and kills an idle worker with
    # "Timeout reading from socket". Every window has to be finite and shorter than
    # the socket timeout — see from_config.
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME)
    redis.on_brpop = lambda: queue.enqueue(_job()) if not redis.brpop_calls else None

    queue.consume()

    assert redis.brpop_calls == [(QUEUE_NAME, BLOCK_SECONDS)]
    assert 0 < BLOCK_SECONDS


def test_configured_socket_timeout_outlives_a_block_window():
    # If the socket gives up first, a perfectly healthy idle BRPOP looks like a
    # network failure. from_url does not connect, so this needs no server.
    kwargs = from_config()._client.connection_pool.connection_kwargs

    assert kwargs["socket_timeout"] > BLOCK_SECONDS
