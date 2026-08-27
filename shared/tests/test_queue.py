import json

import pytest
from pydantic import ValidationError

from shared.models.detection import DetectionJob, FrameRange, JobSource, ViolationType
from shared.models.evidence import EvidenceJob
from shared.queue.client import (
    BLOCK_SECONDS,
    RedisQueue,
    evidence_from_config,
    from_config,
)
from shared.queue.memory import InMemoryQueue

QUEUE_NAME = "detection:jobs"
EVIDENCE_QUEUE_NAME = "evidence:jobs"


def _job(job_id: str = "job-1", start: int = 0, end: int = 100) -> DetectionJob:
    return DetectionJob(
        id=job_id,
        site_id="site-1",
        source=JobSource(
            source_id="source-1", version=3, key="video/file-1/clip.mp4", total_frames=end
        ),
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
    queue = RedisQueue(redis, QUEUE_NAME, DetectionJob)

    queue.enqueue(_job())

    assert json.loads(redis.lists[QUEUE_NAME][0]) == {
        "id": "job-1",
        "site_id": "site-1",
        # The video travels with the job. The worker reads it from here and asks
        # site-service nothing — see JobSource.
        "source": {
            "source_id": "source-1",
            "version": 3,
            "key": "video/file-1/clip.mp4",
            "fps": None,
            "total_frames": 100,
        },
        "frame_range": {"start": 0, "end": 100},
        "types": ["red_light_running"],
        # Versions, not documents. The worker resolves these against the database by
        # (site_id, version), which is what pins a job to the calibration that was
        # active when it was created rather than whatever is active when it runs.
        "calibration_version": None,
        "configuration_version": None,
    }


def test_redis_consume_parses_a_brpop_reply():
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME, DetectionJob)
    job = _job()
    queue.enqueue(job)

    assert queue.consume() == job


def test_redis_queue_is_fifo():
    # LPUSH pushes to the head and BRPOP pops the tail; getting that pair backwards
    # would silently make the queue a stack.
    queue = RedisQueue(FakeRedis(), QUEUE_NAME, DetectionJob)
    for i in range(3):
        queue.enqueue(_job(job_id=f"job-{i}"))

    assert [queue.consume().id for _ in range(3)] == ["job-0", "job-1", "job-2"]


def test_redis_consume_waits_through_empty_windows():
    # An idle worker must survive a quiet queue. consume() keeps waiting rather than
    # returning None, so run()'s "None means drained" only ever fires in memory.
    redis = FakeRedis()
    queue = RedisQueue(redis, QUEUE_NAME, DetectionJob)
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
    queue = RedisQueue(redis, QUEUE_NAME, DetectionJob)
    redis.on_brpop = lambda: queue.enqueue(_job()) if not redis.brpop_calls else None

    queue.consume()

    assert redis.brpop_calls == [(QUEUE_NAME, BLOCK_SECONDS)]
    assert 0 < BLOCK_SECONDS


def test_configured_socket_timeout_outlives_a_block_window():
    # If the socket gives up first, a perfectly healthy idle BRPOP looks like a
    # network failure. from_url does not connect, so this needs no server.
    kwargs = from_config()._client.connection_pool.connection_kwargs

    assert kwargs["socket_timeout"] > BLOCK_SECONDS


# --- the evidence queue ---------------------------------------------------


def test_an_evidence_job_carries_the_violation_and_the_window_it_was_recorded_over():
    # Deliberately thin. A violation row is never rewritten, so unlike DetectionJob
    # nothing reachable from the id could drift between enqueue and consume — and the
    # window it points at runs to tens of kilobytes, which is not a queue payload.
    #
    # evidence_seconds is the exception, and the only one: it lives in the site's
    # configuration document in object storage, so the alternative is an S3 fetch per
    # violation to re-read a number the detector had already resolved.
    redis = FakeRedis()
    queue = RedisQueue(redis, EVIDENCE_QUEUE_NAME, EvidenceJob)

    queue.enqueue(EvidenceJob(violation_id="v-1", evidence_seconds=5.0))

    assert json.loads(redis.lists[EVIDENCE_QUEUE_NAME][0]) == {
        "violation_id": "v-1",
        "evidence_seconds": 5.0,
    }


def test_an_evidence_queue_parses_back_into_an_evidence_job():
    redis = FakeRedis()
    queue = RedisQueue(redis, EVIDENCE_QUEUE_NAME, EvidenceJob)
    job = EvidenceJob(violation_id="v-1", evidence_seconds=5.0)
    queue.enqueue(job)

    assert queue.consume() == job


@pytest.mark.parametrize("seconds", [0, -1])
def test_an_evidence_job_with_no_window_is_not_a_job(seconds):
    # A clip of zero seconds is not a degraded clip, it is a caller who computed the
    # window wrong — and it would reach ffmpeg as a `-t 0` that writes nothing, which
    # surfaces two processes later as a violation whose footage "could not be cut".
    with pytest.raises(ValidationError):
        EvidenceJob(violation_id="v-1", evidence_seconds=seconds)


def test_a_queue_parses_into_the_type_it_was_built_for():
    # The reason the message type is a constructor argument rather than a default. A
    # queue pointed at the wrong list should fail on the payload it cannot read, not
    # quietly hand a caller something of the wrong shape.
    redis = FakeRedis()
    RedisQueue(redis, QUEUE_NAME, DetectionJob).enqueue(_job())
    misdirected = RedisQueue(redis, QUEUE_NAME, EvidenceJob)

    with pytest.raises(ValidationError):
        misdirected.consume()


def test_the_two_queues_are_configured_onto_different_lists():
    # One list for both would make each worker pop messages it cannot handle, and at
    # very different rates: a GPU decoding a video against ffmpeg cutting five seconds.
    assert from_config()._name != evidence_from_config()._name
