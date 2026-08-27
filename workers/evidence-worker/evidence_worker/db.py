"""The worker's handle on the shared database.

The same arrangement detection-worker has, and for the same reasons: one connection
per process, opened before the queue is touched, and the schema checked rather than
created. Running the DDL here would mean a mistyped TRAFFIC_DB_PATH silently produces
an empty database — and this worker would then find no violation for any job and mark
every one of them failed, which looks like broken footage rather than a broken path.

THIS IS THE THIRD PROCESS ON ONE SQLITE FILE, and the second that writes. WAL handles
that between processes on one filesystem, which is exactly the arrangement
docker-compose.yml keeps them in. It is also the thing that stops this worker moving
to its own host — the constraint is the filesystem, not the packaging.
"""

import sqlite3

from shared.config import DB_PATH
from shared.db.connection import get_connection

# Created by site-service at startup. site_sources and files are here because finding
# a violation's footage joins through both — see shared.db.violations.evidence_target.
_REQUIRED_TABLES = ("traffic_violations", "site_sources", "files")

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
