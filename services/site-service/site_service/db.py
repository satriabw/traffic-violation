"""site-service's handle on the shared database.

One connection per thread, not one per process. FastAPI runs sync endpoints — which
is all of them here — in a threadpool, and a sqlite3 connection belongs to the thread
that opened it. A single shared connection raises ProgrammingError from whichever
worker thread happens to pick up the second request, which looks like a load problem
and is really a threading one.

Connections are cheap and the file is WAL, so per-thread costs nothing: readers do not
block each other, and the only serialisation left is between concurrent writers.
"""

import sqlite3
import threading

from shared.config import DB_PATH
from shared.db.connection import get_connection
from shared.db.init import init_db

_local = threading.local()


def init_app_db() -> None:
    """Create the schema, once, at startup.

    Deliberately not the same connection the requests use — this one exists only long
    enough to run the DDL. Every request thread opens its own below, against a file
    that is already migrated.
    """
    con = get_connection(DB_PATH)
    try:
        init_db(con)
    finally:
        con.close()


def get_db() -> sqlite3.Connection:
    con = getattr(_local, "connection", None)
    if con is None:
        con = _local.connection = get_connection(DB_PATH)
    return con
