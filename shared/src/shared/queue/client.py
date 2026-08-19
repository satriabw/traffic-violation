"""The detection job queue, backed by a Redis list.

A list rather than a task framework: this hop needs push and pop, and nothing else
Celery or RQ offers is wanted yet. It also keeps the worker's entrypoint ours rather
than a framework's, which is what makes the consume loop in detection-worker plain
readable Python.

LPUSH at the head, BRPOP from the tail — that pairing is what makes the list FIFO.
Reversing either one turns the queue into a stack, and the oldest job starves.
"""

import redis

from shared import config
from shared.models.detection import DetectionJob


class RedisQueue:
    def __init__(self, client, name: str):
        # The client is injected rather than built here, so tests drive the real
        # commands and the real serialisation against a fake with no server running.
        self._client = client
        self._name = name

    def enqueue(self, job: DetectionJob) -> None:
        self._client.lpush(self._name, job.model_dump_json())

    def consume(self, timeout: int = 0) -> DetectionJob | None:
        """The next job, waiting for one to arrive.

        Returns None only when `timeout` elapses first. The default of 0 means BRPOP
        blocks indefinitely, which is what a worker wants — polling would cost a round
        trip per interval to learn nothing.
        """
        reply = self._client.brpop(self._name, timeout)
        if reply is None:
            return None
        _, payload = reply
        return DetectionJob.model_validate_json(payload)


def from_config() -> RedisQueue:
    """The queue this deployment is configured for."""
    # decode_responses so payloads come back as str: model_validate_json accepts
    # bytes too, but every other reader would have to remember to decode.
    return RedisQueue(
        redis.Redis.from_url(config.REDIS_URL, decode_responses=True),
        config.DETECTION_QUEUE_NAME,
    )
