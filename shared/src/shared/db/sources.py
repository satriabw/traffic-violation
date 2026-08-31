"""Writing a source's status, from either side of the queue.

Here rather than in site-service for the reason `shared.db.violations` is: two processes
write this column now — site-service creates the row, detection-worker moves it through
the run — and one copy of the statements is the only way they stay in step with the
schema they are written against.

WHAT THE STATUS IS FOR. Detection is asynchronous, so between asking for it and the
first violation appearing there is a stretch where the site's violation list is empty —
and an empty list is exactly what a site with no violations returns. Without this the
two are indistinguishable, and a user watching a run in progress is told there is
nothing to see. The column has carried the states to say otherwise since the schema was
written; nothing had ever set them.
"""

import sqlite3

from shared.models.source import SourceStatus


def set_source_status(
    con: sqlite3.Connection, source_id: str, status: SourceStatus
) -> None:
    """Move one source to a new status.

    No check that the transition makes sense. The CHECK constraint on the column says
    which values exist, the worker is the only thing that writes the video states, and a
    state machine enforced here would be a second opinion about an order that already
    only has one writer.

    `updated_at` is bumped by hand, the same way set_evidence and set_explanation do it:
    the column defaults on insert and SQLite has no ON UPDATE, so a row that changed
    without this would keep claiming the time it was created.
    """
    con.execute(
        """
        UPDATE site_sources
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        [status.value, source_id],
    )


def fail_processing_sources(con: sqlite3.Connection) -> int:
    """Mark every source still mid-analysis as failed. Returns how many.

    Run once when detection-worker starts. A worker killed part-way through a job leaves
    its source on 'processing' and no longer exists to move it off — the row goes on
    claiming an analysis is running, and anything reading it shows a user "analysing…"
    for good. Failed is the true statement and the one they can act on.

    IT ASSUMES ONE DETECTION-WORKER, and that assumption is the whole of its
    correctness — the same one `fail_pending_explanations` makes about site-service. A
    second worker starting cannot tell a source abandoned by a dead process from one a
    live sibling is part-way through, and would mark a running analysis failed. What
    makes that safe is what makes the deployment single-writer today; a second worker
    needs a claim on the row naming who holds it, and that is what would make this safe
    too.

    STREAM STATES ARE NOT TOUCHED, and the WHERE clause is what keeps it that way.
    'active' and 'degraded' describe a live feed rather than a run over a video, so
    nothing here has any business deciding they are over.
    """
    return con.execute(
        """
        UPDATE site_sources
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE status = ?
        """,
        [SourceStatus.FAILED.value, SourceStatus.PROCESSING.value],
    ).rowcount
