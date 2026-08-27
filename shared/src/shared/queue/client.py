"""Job queues, backed by Redis lists.

A list rather than a task framework: this hop needs push and pop, and nothing else
Celery or RQ offers is wanted yet. It also keeps each worker's entrypoint ours rather
than a framework's, which is what makes the consume loops plain readable Python.

LPUSH at the head, BRPOP from the tail — that pairing is what makes the list FIFO.
Reversing either one turns the queue into a stack, and the oldest job starves.

TWO LISTS, ONE CLASS. Detection jobs and evidence jobs travel the same way and differ
only in what they deserialise to, so the message type is a constructor argument rather
than a second class. It is required rather than defaulted: a queue that guessed
DetectionJob would parse an evidence job into a validation error at the far end of a
network hop, which is the worst place to discover a queue was pointed at the wrong
list.
"""

from typing import Generic, Type, TypeVar

import redis
from pydantic import BaseModel

from shared import config
from shared.models.detection import DetectionJob
from shared.models.evidence import EvidenceJob

Message = TypeVar("Message", bound=BaseModel)

# How long one BRPOP waits before we issue another. It must stay *below* the
# connection's socket timeout: BRPOP with 0 ("wait forever") outlives redis-py's
# 5-second default and an idle worker dies with "Timeout reading from socket", which
# looks like a network fault and is really just a quiet queue.
BLOCK_SECONDS = 5

# Slack over BLOCK_SECONDS so the server always answers first. Finite rather than
# None so a genuinely dead connection still surfaces instead of hanging forever.
SOCKET_TIMEOUT_SECONDS = BLOCK_SECONDS * 2


class RedisQueue(Generic[Message]):
    def __init__(self, client, name: str, message: Type[Message]):
        # The client is injected rather than built here, so tests drive the real
        # commands and the real serialisation against a fake with no server running.
        self._client = client
        self._name = name
        self._message = message

    def enqueue(self, job: Message) -> None:
        self._client.lpush(self._name, job.model_dump_json())

    def consume(self) -> Message:
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
                return self._message.model_validate_json(payload)


def _queue(name: str, message: Type[Message]) -> RedisQueue[Message]:
    # decode_responses so payloads come back as str: model_validate_json accepts
    # bytes too, but every other reader would have to remember to decode.
    return RedisQueue(
        redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
        ),
        name,
        message,
    )


def from_config() -> RedisQueue[DetectionJob]:
    """The detection queue this deployment is configured for.

    Keeps its bare name because it is the older of the two and every one of its callers
    predates the second — renaming it would touch site-service and detection-worker to
    say something they already say by importing it.
    """
    return _queue(config.DETECTION_QUEUE_NAME, DetectionJob)


def evidence_from_config() -> RedisQueue[EvidenceJob]:
    """The evidence queue this deployment is configured for.

    A separate connection from the detection queue's, even where one process holds
    both. redis-py pools per client, and the two are only ever used from the same
    thread — sharing one would save a socket and couple the lifetimes of two things
    that fail for different reasons.
    """
    return _queue(config.EVIDENCE_QUEUE_NAME, EvidenceJob)
