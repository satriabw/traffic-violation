"""Recording a violation.

The only thing in this process that writes. Reads — a site's calibration and
configuration — resolve by version and are handled elsewhere.
"""

import json
import sqlite3
import uuid

from shared.models.violation import ViolationCreate


def record(con: sqlite3.Connection, violation: ViolationCreate) -> str:
    """Write the violation and its metadata, and return the new id.

    Evidence frames must already be in S3 before this is called. That ordering is the
    LLD's and it is the right way round: an orphaned frame in object storage is
    harmless, while a violation row pointing at frames that were never uploaded is
    broken and stays broken.

    The two inserts share one transaction. Connections here are autocommit, so the
    BEGIN is explicit — without it a crash between the statements leaves a violation
    with no trajectories behind it, which reads as a detection nobody can review.
    BEGIN IMMEDIATE rather than plain BEGIN so the write lock is taken up front and
    two writers contend at the start rather than half way through.
    """
    violation_id = str(uuid.uuid4())
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            """
            INSERT INTO traffic_violations (id, site_id, type, detected_at)
            VALUES (?, ?, ?, ?)
            """,
            [violation_id, violation.site_id, violation.type.value, violation.detected_at],
        )
        con.execute(
            """
            INSERT INTO violation_metadata (id, traffic_violation_id, json_blob)
            VALUES (?, ?, ?)
            """,
            [str(uuid.uuid4()), violation_id, violation.metadata.model_dump_json()],
        )
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    return violation_id
