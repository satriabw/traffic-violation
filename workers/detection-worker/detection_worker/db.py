"""The worker's handle on the shared database.

New in this worker, and a deliberate change of position: until now it deliberately
had no database and no HTTP client, because everything a job needed travelled in the
message. Violations broke that — a detection has to be recorded somewhere, and a
queue is not a store.

What has *not* changed is that the worker never calls site-service. It reads the same
file, which is a coupling at the storage layer rather than a runtime dependency: a
backlog still drains while site-service is down, which was the point.

One connection per process. The consume loop is single-threaded, so there is nothing
to share it with.
"""

import sqlite3

from shared.config import DB_PATH
from shared.db.connection import get_connection

# Created by site-service at startup. The worker checks rather than creates: running
# the DDL here would mean a mistyped TRAFFIC_DB_PATH silently produces an empty
# database, and the worker would then find no calibration for any site and record
# nothing, looking like quiet footage rather than a misconfiguration.
_REQUIRED_TABLES = ("sites", "traffic_violations", "violation_metadata")

_connection: sqlite3.Connection | None = None


class SchemaMissing(RuntimeError):
    pass


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open the shared database, refusing one that has no schema.

    `db_path` is injectable so tests point at a temp file; nothing else passes it.
    """
    con = get_connection(db_path or DB_PATH)
    present = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing = [table for table in _REQUIRED_TABLES if table not in present]
    if missing:
        con.close()
        raise SchemaMissing(
            f"{db_path or DB_PATH} has no {', '.join(missing)} table. "
            "site-service creates the schema at startup — check TRAFFIC_DB_PATH points "
            "at the same file, and that site-service has run at least once."
        )
    return con


def get_db() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = connect()
    return _connection
