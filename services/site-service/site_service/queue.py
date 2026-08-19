"""Detection queue dependency.

Injected the same way get_storage and get_probe are, so tests substitute an
InMemoryQueue through app.dependency_overrides — and so no test can reach a real Redis.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from shared.queue.client import from_config


@lru_cache
def get_queue():
    # Built lazily (first request) and cached for the process lifetime so the
    # underlying redis connection pool is reused across requests.
    return from_config()


# The concrete type is RedisQueue or InMemoryQueue; routers depend on the shape
# (enqueue/consume), not the identity.
Queue = Annotated[object, Depends(get_queue)]
