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

# How long one BRPOP waits before we issue another. It must stay *below* the
# connection's socket timeout: BRPOP with 0 ("wait forever") outlives redis-py's
# 5-second default and an idle worker dies with "Timeout reading from socket", which
# looks like a network fault and is really just a quiet queue.
BLOCK_SECONDS = 5

# Slack over BLOCK_SECONDS so the server always answers first. Finite rather than
# None so a genuinely dead connection still surfaces instead of hanging forever.
SOCKET_TIMEOUT_SECONDS = BLOCK_SECONDS * 2


class RedisQueue:
    def __init__(self, client, name: str):
        # The client is injected rather than built here, so tests drive the real
        # commands and the real serialisation against a fake with no server running.
        self._client = client
        self._name = name

    def enqueue(self, job: DetectionJob) -> None:
        self._client.lpush(self._name, job.model_dump_json())

    def consume(self) -> DetectionJob:
        """The next job, waiting as long as it takes for one to arrive.

        Never returns None: an empty queue means "not yet", not "stop". That is what
        keeps `None` in the worker loop meaning only what InMemoryQueue means by it —
        drained, nothing more is coming.

        The waiting is a series of bounded BRPOPs rather than one unbounded one. Each
        window costs a single round trip every BLOCK_SECONDS, so this is still
        blocking rather than polling, but the connection is never idle long enough for
        the socket timeout to fire.
        """
        while True:
            reply = self._client.brpop(self._name, BLOCK_SECONDS)
            if reply is not None:
                _, payload = reply
                return DetectionJob.model_validate_json(payload)


def from_config() -> RedisQueue:
    """The queue this deployment is configured for."""
    # decode_responses so payloads come back as str: model_validate_json accepts
    # bytes too, but every other reader would have to remember to decode.
    return RedisQueue(
        redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
        ),
        config.DETECTION_QUEUE_NAME,
    )
