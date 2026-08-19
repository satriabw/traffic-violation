"""Detection queue dependency.

Injected the same way get_storage and get_probe are, so tests substitute an
InMemoryQueue through app.dependency_overrides — and so no test can reach a real Redis.
"""

from typing import Annotated

from fastapi import Depends

from shared.queue.client import from_config


def get_queue():
    # Built per request rather than at import: redis-py pools connections internally,
    # and constructing at import time would make a module import depend on config
    # having been loaded.
    return from_config()


# The concrete type is RedisQueue or InMemoryQueue; routers depend on the shape
# (enqueue/consume), not the identity.
Queue = Annotated[object, Depends(get_queue)]
