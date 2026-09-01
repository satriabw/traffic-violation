"""Moving a source through a detection run.

The column and its permitted values have been in the schema since it was written; until
now nothing set the three that describe a video being analysed. These are the writes
that make an empty violation list mean something a user can be told.
"""

import sqlite3

import pytest

from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.sources import fail_processing_sources, set_source_status
from shared.models.source import SourceStatus


def _db():
    con = get_connection(":memory:")
    init_db(con)
    con.execute("INSERT INTO sites (id, name) VALUES ('site-1', 'Junction 5')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status) "
        "VALUES ('file-1', 'a.mp4', 'video/f/a.mp4', 'video', 'uploaded')"
    )
    return con


def _source(con, source_id: str, version: int, status: str | None = None) -> str:
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id) "
        "VALUES (?, 'site-1', ?, 'video', 'file-1')",
        [source_id, version],
    )
    if status is not None:
        con.execute("UPDATE site_sources SET status = ? WHERE id = ?", [status, source_id])
    return source_id


def _status(con, source_id: str) -> str:
    return con.execute(
        "SELECT status FROM site_sources WHERE id = ?", [source_id]
    ).fetchone()[0]


def test_a_new_source_starts_out_having_had_nothing_done_to_it():
    con = _db()

    assert _status(con, _source(con, "src-1", 1)) == SourceStatus.CREATED.value


def test_a_source_can_be_moved_through_a_run():
    con = _db()
    source_id = _source(con, "src-1", 1)

    set_source_status(con, source_id, SourceStatus.PROCESSING)
    assert _status(con, source_id) == SourceStatus.PROCESSING.value

    set_source_status(con, source_id, SourceStatus.COMPLETED)
    assert _status(con, source_id) == SourceStatus.COMPLETED.value


def test_the_column_still_refuses_a_status_nobody_defined():
    # The CHECK is what says which values exist; set_source_status does not second-guess
    # it, so this is where an unknown one is caught.
    con = _db()
    source_id = _source(con, "src-1", 1)

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "UPDATE site_sources SET status = 'banana' WHERE id = ?", [source_id]
        )


def test_writing_a_status_bumps_updated_at():
    # The column defaults on insert and SQLite has no ON UPDATE, so a row that changed
    # without this would keep claiming the time it was created.
    con = _db()
    source_id = _source(con, "src-1", 1)
    con.execute(
        "UPDATE site_sources SET updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        [source_id],
    )

    set_source_status(con, source_id, SourceStatus.PROCESSING)

    updated = con.execute(
        "SELECT updated_at FROM site_sources WHERE id = ?", [source_id]
    ).fetchone()[0]
    assert str(updated) > "2020-01-01"


def test_a_source_stranded_mid_analysis_is_failed_at_startup():
    """The worker that was analysing it is gone and will not be moving it on.

    Without this the row goes on claiming an analysis is running, and anything reading
    it shows a user "analysing..." for good.
    """
    con = _db()
    source_id = _source(con, "src-1", 1, status=SourceStatus.PROCESSING.value)

    assert fail_processing_sources(con) == 1

    assert _status(con, source_id) == SourceStatus.FAILED.value


def test_the_sweep_leaves_every_other_status_alone():
    con = _db()
    untouched = {
        "created": _source(con, "src-created", 1),
        "completed": _source(con, "src-done", 2, status=SourceStatus.COMPLETED.value),
        "failed": _source(con, "src-failed", 3, status=SourceStatus.FAILED.value),
        # Stream states. They describe a live feed rather than a run over a video, so
        # nothing about a worker restarting has any business ending them.
        "active": _source(con, "src-active", 4, status=SourceStatus.ACTIVE.value),
        "degraded": _source(con, "src-degraded", 5, status=SourceStatus.DEGRADED.value),
    }

    assert fail_processing_sources(con) == 0

    for expected, source_id in untouched.items():
        assert _status(con, source_id) == expected


def test_the_sweep_reports_how_many_it_moved():
    con = _db()
    _source(con, "src-1", 1, status=SourceStatus.PROCESSING.value)
    _source(con, "src-2", 2, status=SourceStatus.PROCESSING.value)
    _source(con, "src-3", 3)

    assert fail_processing_sources(con) == 2


def test_the_sweep_is_a_no_op_on_a_database_with_nothing_running():
    con = _db()
    _source(con, "src-1", 1)

    assert fail_processing_sources(con) == 0
