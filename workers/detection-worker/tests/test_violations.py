import sqlite3
from datetime import datetime, timezone

import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.models.detection import ViolationType
from shared.models.violation import TrackSummary, ViolationCreate, ViolationMetadata

from detection_worker.violations import record

DETECTED_AT = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def con():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    connection.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'junction.mp4', 'video/f1/junction.mp4', 'video', 'uploaded')"
    )
    connection.execute(
        """
        INSERT INTO site_sources (id, site_id, version, kind, file_id)
        VALUES ('src-1', 's1', 3, 'video', 'f1')
        """
    )
    return connection


def _violation(**overrides) -> ViolationCreate:
    return ViolationCreate(
        **{
            "site_id": "s1",
            "source_id": "src-1",
            "frame_index": 912,
            "type": ViolationType.RED_LIGHT_RUNNING,
            "detected_at": DETECTED_AT,
            "metadata": ViolationMetadata(
                vehicles=[
                    TrackSummary(
                        track_id=7,
                        trajectory=[(1.0, 2.0), (1.5, 2.5)],
                        speed=[11.0, 12.0],
                        frame_idxs=[100, 101],
                        bboxes=[(0, 0, 10, 10), (1, 1, 11, 11)],
                    )
                ],

            ),
            **overrides,
        }
    )


def test_record_writes_the_violation_and_its_metadata(con):
    violation_id = record(con, _violation())

    row = con.execute(
        "SELECT site_id, type, status, detected_at FROM traffic_violations WHERE id = ?",
        [violation_id],
    ).fetchone()
    assert row == ("s1", "red_light_running", "detected", DETECTED_AT)
    assert (
        con.execute(
            "SELECT COUNT(*) FROM violation_metadata WHERE traffic_violation_id = ?",
            [violation_id],
        ).fetchone()[0]
        == 1
    )


def test_recorded_metadata_reads_back_as_the_model_that_was_written(con):
    original = _violation()
    violation_id = record(con, original)

    blob = con.execute(
        "SELECT json_blob FROM violation_metadata WHERE traffic_violation_id = ?",
        [violation_id],
    ).fetchone()[0]
    assert ViolationMetadata.model_validate_json(blob) == original.metadata


def test_detected_at_survives_the_round_trip_as_an_aware_datetime(con):
    """The worker records when it happened in the footage, and a reviewer compares
    that against the video. A timestamp that came back naive, or shifted, would make
    every violation point at the wrong moment."""
    violation_id = record(con, _violation())

    stored = con.execute(
        "SELECT detected_at FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()[0]
    assert stored == DETECTED_AT
    assert stored.tzinfo is not None


def test_a_violation_for_an_unknown_site_leaves_nothing_behind(con):
    """Both inserts share a transaction, so a rejected violation must not leave a
    metadata row orphaned — nor a violation with no trajectories, which reads as a
    detection nobody can review."""
    with pytest.raises(sqlite3.IntegrityError):
        record(con, _violation(site_id="no-such-site"))

    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM violation_metadata").fetchone()[0] == 0
    # The failed transaction must also have been rolled back, not left open — the next
    # write would otherwise fail with "cannot start a transaction within a transaction".
    record(con, _violation())
    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 1


def test_a_violation_records_which_video_it_came_from_and_where(con):
    """The whole reason no frames are uploaded. Without these two a violation cannot
    say what to open or where to seek, and the evidence can never be re-derived."""
    violation_id = record(con, _violation())

    assert con.execute(
        "SELECT source_id, frame_index FROM traffic_violations WHERE id = ?",
        [violation_id],
    ).fetchone() == ("src-1", 912)


def test_the_pinned_source_is_the_version_the_job_ran_against(con):
    """site_sources appends a row per version and its id is the primary key, so the
    id pins the version on its own — no second column to disagree with it."""
    con.execute(
        """
        INSERT INTO site_sources (id, site_id, version, kind, file_id)
        VALUES ('src-2', 's1', 4, 'video', 'f1')
        """
    )
    violation_id = record(con, _violation())

    assert con.execute(
        """
        SELECT s.version FROM traffic_violations v
        JOIN site_sources s ON s.id = v.source_id
        WHERE v.id = ?
        """,
        [violation_id],
    ).fetchone()[0] == 3


def test_a_violation_naming_a_source_that_does_not_exist_leaves_nothing_behind(con):
    """A row that cannot locate its own footage is a detection nobody can review, so
    it is refused rather than written and puzzled over later."""
    with pytest.raises(sqlite3.IntegrityError):
        record(con, _violation(source_id="no-such-source"))

    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM violation_metadata").fetchone()[0] == 0


def test_no_evidence_frames_are_recorded(con):
    """Settled, not pending: the pixels come back from the source on demand."""
    violation_id = record(con, _violation())

    blob = con.execute(
        "SELECT json_blob FROM violation_metadata WHERE traffic_violation_id = ?",
        [violation_id],
    ).fetchone()[0]
    assert ViolationMetadata.model_validate_json(blob).frames == []
