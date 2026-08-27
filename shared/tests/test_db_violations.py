import json

import pytest

from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import EvidenceTarget, evidence_target, record, set_evidence
from shared.models.detection import ViolationType
from shared.models.violation import EvidenceStatus, ViolationCreate

from datetime import datetime, timezone

DETECTED_AT = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def con():
    con = get_connection(":memory:")
    init_db(con)
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'a.mp4', 'video/f1/a.mp4', 'video', 'uploaded')"
    )
    return con


def _source(con, metadata=None, source_id="src1", version=1):
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id, metadata)"
        " VALUES (?, 's1', ?, 'video', 'f1', ?)",
        [source_id, version, metadata],
    )
    return source_id


def _violation(con, source_id="src1", frame_index=912):
    return record(
        con,
        ViolationCreate(
            site_id="s1",
            source_id=source_id,
            frame_index=frame_index,
            type=ViolationType.RED_LIGHT_RUNNING,
            detected_at=DETECTED_AT,
        ),
    )


# --- what record leaves behind ------------------------------------------------


def test_a_recorded_violation_has_no_evidence_and_no_job_waiting_on_one(con):
    # NULL rather than 'pending'. The detector knows a rule fired; whether anything has
    # been queued to cut the footage is not its fact to state, and a row claiming
    # 'pending' with nothing enqueued is a reader waiting for good.
    _source(con)

    violation_id = _violation(con)

    assert con.execute(
        "SELECT thumbnail_key, clip_key, evidence_status FROM traffic_violations"
        " WHERE id = ?",
        [violation_id],
    ).fetchone() == (None, None, None)


# --- finding the footage ------------------------------------------------------


def test_a_violation_points_at_the_video_and_the_frame_it_happened_on(con):
    _source(con, metadata=json.dumps({"fps": 29.97, "total_frames": 5400}))
    violation_id = _violation(con, frame_index=912)

    assert evidence_target(con, violation_id) == EvidenceTarget(
        key="video/f1/a.mp4", frame_index=912, fps=29.97
    )


def test_the_video_is_the_version_the_violation_was_pinned_to(con):
    # Not the site's active source. A violation found in v1 is still a violation found
    # in v1 after v2 is attached, and cutting the clip out of the newer video would
    # produce evidence of something that never happened.
    _source(con, metadata=json.dumps({"fps": 30.0}))
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f2', 'b.mp4', 'video/f2/b.mp4', 'video', 'uploaded')"
    )
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id)"
        " VALUES ('src2', 's1', 2, 'video', 'f2')"
    )
    violation_id = _violation(con, source_id="src1")

    assert evidence_target(con, violation_id).key == "video/f1/a.mp4"


def test_a_violation_that_does_not_exist_has_no_footage(con):
    assert evidence_target(con, "no-such-violation") is None


def test_a_violation_recorded_before_the_source_columns_cannot_be_cut(con):
    # It genuinely does not know which video it came from. Indistinguishable here from
    # a violation that is not there at all, and deliberately so — neither can ever be
    # cut, so the caller does the same thing for both.
    con.execute(
        "INSERT INTO traffic_violations (id, site_id, type, detected_at)"
        " VALUES ('v-old', 's1', 'red_light_running', '2026-08-21 10:00:00')"
    )

    assert evidence_target(con, "v-old") is None


@pytest.mark.parametrize(
    "label,metadata",
    [
        ("no metadata document at all", None),
        ("a document that never had a frame rate", json.dumps({"total_frames": 5400})),
        ("a frame rate the probe could not read", json.dumps({"fps": None})),
        ("a document that is not json", "{not json"),
        ("a frame rate that is not a number", json.dumps({"fps": "thirty"})),
    ],
)
def test_a_source_whose_frame_rate_is_unknown_reports_none_rather_than_guessing(
    con, label, metadata
):
    # A guessed frame rate seeks to the wrong second of a real video, and the clip that
    # comes back looks like a detector that fired at nothing. The target still resolves
    # — the key and the frame index are known — and the caller decides what to do.
    _source(con, metadata=metadata)
    violation_id = _violation(con)

    target = evidence_target(con, violation_id)

    assert target.key == "video/f1/a.mp4"
    assert target.fps is None


# --- writing the result back --------------------------------------------------


def test_a_finished_cut_puts_both_keys_on_the_row(con):
    _source(con)
    violation_id = _violation(con)

    set_evidence(
        con,
        violation_id,
        EvidenceStatus.READY,
        thumbnail_key="evidence/v1/thumbnail.jpg",
        clip_key="evidence/v1/clip.mp4",
    )

    assert con.execute(
        "SELECT evidence_status, thumbnail_key, clip_key FROM traffic_violations"
        " WHERE id = ?",
        [violation_id],
    ).fetchone() == ("ready", "evidence/v1/thumbnail.jpg", "evidence/v1/clip.mp4")


def test_a_failed_cut_does_not_leave_the_previous_attempt_s_keys_behind(con):
    # The keys are written on every call, Nones included. A row that kept the earlier
    # attempt's key alongside a 'failed' status points at an object that may since have
    # been overwritten, which is worse than pointing at nothing.
    _source(con)
    violation_id = _violation(con)
    set_evidence(
        con,
        violation_id,
        EvidenceStatus.READY,
        thumbnail_key="evidence/v1/thumbnail.jpg",
        clip_key="evidence/v1/clip.mp4",
    )

    set_evidence(con, violation_id, EvidenceStatus.FAILED)

    assert con.execute(
        "SELECT evidence_status, thumbnail_key, clip_key FROM traffic_violations"
        " WHERE id = ?",
        [violation_id],
    ).fetchone() == ("failed", None, None)


def test_writing_evidence_touches_the_row_s_updated_at(con):
    # DEFAULT CURRENT_TIMESTAMP fires on the INSERT and never again, so without the
    # explicit bump a violation would claim nothing had happened to it since detection.
    _source(con)
    violation_id = _violation(con)
    con.execute(
        "UPDATE traffic_violations SET updated_at = '2000-01-01 00:00:00' WHERE id = ?",
        [violation_id],
    )

    set_evidence(con, violation_id, EvidenceStatus.PENDING)

    updated_at = con.execute(
        "SELECT updated_at FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()[0]
    assert updated_at.year > 2000


def test_evidence_written_for_one_violation_leaves_the_others_alone(con):
    _source(con)
    mine = _violation(con, frame_index=100)
    theirs = _violation(con, frame_index=200)

    set_evidence(con, mine, EvidenceStatus.READY, thumbnail_key="k", clip_key="c")

    assert con.execute(
        "SELECT evidence_status FROM traffic_violations WHERE id = ?", [theirs]
    ).fetchone()[0] is None
