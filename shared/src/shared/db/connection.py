"""The one database connection, opened the same way by every process that needs it.

SQLite rather than DuckDB because this file now has more than one process on it.
DuckDB allows a single read-write process *or* several read-only ones, never both, so
detection-worker could not open the file at all while site-service held it. SQLite in
WAL mode gives concurrent readers alongside one writer, which is exactly the shape of
this system: an API taking human-rate writes and a worker appending violations.

That only holds while the file is on a local filesystem shared by every process. A
worker on its own host, or replicas across machines, is where this stops working and
Postgres starts — the trigger is the filesystem, not the row count.
"""

import datetime
import os
import sqlite3

# Python 3.12 deprecated the built-in datetime adapter and timestamp converter, and CI
# runs 3.13, so both directions are ours. Registered as a pair on purpose: the format
# written has to be the one read back, and the separator below is a space so a stored
# datetime is byte-identical to what CURRENT_TIMESTAMP produces for the columns that
# default to it. fromisoformat accepts both that and the "T" form, offset or not.
sqlite3.register_adapter(datetime.datetime, lambda value: value.isoformat(sep=" "))
sqlite3.register_converter(
    "TIMESTAMP", lambda raw: datetime.datetime.fromisoformat(raw.decode())
)

# How long a writer waits for another writer's lock before giving up. WAL keeps
# readers out of this entirely; only two concurrent *writers* ever reach it, and a
# violation insert is far shorter than this.
BUSY_TIMEOUT_MS = 5000


def get_connection(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        parent = os.path.dirname(db_path)
        # Empty for a bare filename, which is already the current directory.
        if parent:
            os.makedirs(parent, exist_ok=True)

    con = sqlite3.connect(
        db_path,
        # Pairs with the converter above, so callers get datetimes rather than strings.
        detect_types=sqlite3.PARSE_DECLTYPES,
        # Autocommit. sqlite3 otherwise opens an implicit transaction before the first
        # write and holds it until someone commits, which would turn every
        # read-then-insert in the service layer into a lock held across a round trip.
        isolation_level=None,
        # Connections here legitimately cross threads: FastAPI runs sync endpoints on a
        # threadpool, and the tests hand one :memory: connection to every request. Safe
        # because this build reports sqlite3.threadsafety == 3 (serialized) — SQLite
        # holds its own mutex, and the check being disabled is Python's, not SQLite's.
        check_same_thread=False,
    )
    # Off by default in SQLite, per connection — without this line every REFERENCES
    # clause in init.py is decorative and a child row can outlive its parent.
    con.execute("PRAGMA foreign_keys = ON")
    con.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    # The reason this module exists. Persistent in the file once set, but set on every
    # connection so a fresh database gets it before anyone writes. A :memory: database
    # has no WAL and quietly stays in "memory" mode, which is correct for tests.
    con.execute("PRAGMA journal_mode = WAL")
    return con
