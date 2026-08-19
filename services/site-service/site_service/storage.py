"""Storage dependency.

Injected the same way get_db is, so tests substitute a fake through
app.dependency_overrides instead of monkeypatching module internals — and so no test
can accidentally reach R2.
"""

from shared.s3 import client


def get_storage():
    return client
