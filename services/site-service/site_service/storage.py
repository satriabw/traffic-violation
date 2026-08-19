"""Storage dependency.

Injected the same way get_db is, so tests substitute a fake through
app.dependency_overrides instead of monkeypatching module internals — and so no test
can accidentally reach R2.
"""

from typing import Annotated

from fastapi import Depends

from shared.s3 import client


def get_storage():
    return client


# The concrete type is the shared.s3.client module, which has no nominal type to name;
# routers depend on the shape, not the identity.
Storage = Annotated[object, Depends(get_storage)]
