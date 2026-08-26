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

    No evidence frames are uploaded, here or anywhere. The row pins the source and the
    frame index, and the source is an immutable object in storage that can be seeked
    back into, so the pixels are re-derived when somebody opens the detail view — by
    whatever knows how to draw them then, rather than by this process guessing now.
    `ViolationMetadata.frames` stays empty, and the ordering problem the LLD worried
    about (a row pointing at frames nobody uploaded) cannot arise.

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
            INSERT INTO traffic_violations
                (id, site_id, source_id, frame_index, type, detected_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                violation_id,
                violation.site_id,
                violation.source_id,
                violation.frame_index,
                violation.type.value,
                violation.detected_at,
            ],
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
