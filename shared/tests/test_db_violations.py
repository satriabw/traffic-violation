import json

import pytest

from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import (
    EvidenceTarget,
    evidence_target,
    get_with_metadata,
    list_for_setup,
    record,
    set_evidence,
    set_explanation,
)
from shared.models.detection import ViolationType
from shared.models.violation import (
    EvidenceStatus,
    EvidenceStrength,
    LicensePlateAssessment,
    PlateRecoverability,
    Severity,
    TrackSummary,
    ViolationCreate,
    ViolationExplanation,
    ViolationMetadata,
    ViolationStatus,
)

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


def _violation(
    con,
    source_id="src1",
    frame_index=912,
    site_id="s1",
    calibration_id=None,
    configuration_id=None,
    detected_at=DETECTED_AT,
):
    return record(
        con,
        ViolationCreate(
            site_id=site_id,
            source_id=source_id,
            frame_index=frame_index,
            calibration_id=calibration_id,
            configuration_id=configuration_id,
            type=ViolationType.RED_LIGHT_RUNNING,
            detected_at=detected_at,
        ),
    )


def _calibration(con, doc_id, version=1, site_id="s1"):
    con.execute(
        "INSERT INTO camera_calibrations (id, site_id, file_id, version)"
        " VALUES (?, ?, 'f1', ?)",
        [doc_id, site_id, version],
    )
    return doc_id


def _configuration(con, doc_id, version=1, site_id="s1"):
    con.execute(
        "INSERT INTO configurations (id, site_id, file_id, version) VALUES (?, ?, 'f1', ?)",
        [doc_id, site_id, version],
    )
    return doc_id


def _ids(page):
    return [item["id"] for item in page]


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


# --- listing a site's violations under one setup ------------------------------


def test_the_list_is_scoped_to_the_setup_the_violations_were_judged_against(con):
    # The whole point of the two id columns. A site re-calibrated yesterday still holds
    # every violation found under the old camera model, and drawing those with today's
    # polygons puts the vehicle outside the box it was convicted in.
    _source(con)
    old_cal, new_cal = _calibration(con, "cal1"), _calibration(con, "cal2", version=2)
    cfg = _configuration(con, "cfg1")
    under_old = _violation(con, calibration_id=old_cal, configuration_id=cfg)
    _violation(con, calibration_id=new_cal, configuration_id=cfg)

    page, total = list_for_setup(con, "s1", old_cal, cfg, limit=10, offset=0)

    assert _ids(page) == [under_old]
    assert total == 1


def test_a_site_with_no_calibration_finds_the_violations_that_had_none(con):
    # `calibration_id = NULL` is NULL, never true, so an `=` here would return an empty
    # page for a site that is running perfectly well without a camera model. This test
    # fails on `=` and passes on `IS`.
    _source(con)
    cfg = _configuration(con, "cfg1")
    without = _violation(con, calibration_id=None, configuration_id=cfg)

    page, total = list_for_setup(con, "s1", None, cfg, limit=10, offset=0)

    assert _ids(page) == [without]
    assert total == 1


def test_a_violation_judged_under_a_calibration_is_not_returned_for_a_site_with_none(con):
    # The other direction of the same clause: null-safe matching must still exclude,
    # not wave everything through.
    _source(con)
    cal, cfg = _calibration(con, "cal1"), _configuration(con, "cfg1")
    _violation(con, calibration_id=cal, configuration_id=cfg)

    assert list_for_setup(con, "s1", None, cfg, limit=10, offset=0) == ([], 0)


def test_one_site_never_sees_another_sites_violations(con):
    con.execute("INSERT INTO sites (id, name) VALUES ('s2', 'Junction 6')")
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id)"
        " VALUES ('src2', 's2', 1, 'video', 'f1')"
    )
    mine = _violation(con, source_id=_source(con))
    _violation(con, site_id="s2", source_id="src2")

    page, total = list_for_setup(con, "s1", None, None, limit=10, offset=0)

    assert _ids(page) == [mine]
    assert total == 1


def test_the_newest_violation_comes_first(con):
    _source(con)
    older = _violation(con, detected_at=datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc))
    newer = _violation(con, detected_at=datetime(2026, 8, 21, 11, 0, tzinfo=timezone.utc))

    page, _ = list_for_setup(con, "s1", None, None, limit=10, offset=0)

    assert _ids(page) == [newer, older]


def test_violations_sharing_a_moment_do_not_repeat_or_vanish_across_pages(con):
    # One frame can fire a rule for more than one track, so a tie on detected_at is
    # ordinary rather than exotic. Without the id tiebreak SQLite may order the tied
    # rows differently per statement, and two pages read that way overlap.
    _source(con)
    recorded = {_violation(con) for _ in range(4)}

    first, total = list_for_setup(con, "s1", None, None, limit=2, offset=0)
    second, _ = list_for_setup(con, "s1", None, None, limit=2, offset=2)

    assert total == 4
    assert set(_ids(first)) | set(_ids(second)) == recorded
    assert not set(_ids(first)) & set(_ids(second))


def test_the_total_counts_the_whole_filter_not_the_page(con):
    _source(con)
    for _ in range(3):
        _violation(con)

    page, total = list_for_setup(con, "s1", None, None, limit=1, offset=0)

    assert len(page) == 1
    assert total == 3


def test_the_list_carries_the_evidence_keys_without_reaching_for_the_blob(con):
    # The reason the keys are columns. A page of violations has to render a thumbnail
    # each, and violation_metadata is a separate table precisely so the list never
    # touches it.
    _source(con)
    violation_id = _violation(con)
    set_evidence(
        con,
        violation_id,
        EvidenceStatus.READY,
        thumbnail_key="evidence/v/thumbnail.jpg",
        clip_key="evidence/v/clip.mp4",
    )

    (item,), _ = list_for_setup(con, "s1", None, None, limit=10, offset=0)

    assert item["thumbnail_key"] == "evidence/v/thumbnail.jpg"
    assert item["clip_key"] == "evidence/v/clip.mp4"
    assert item["evidence_status"] == EvidenceStatus.READY.value
    assert "metadata" not in item and "json_blob" not in item


def test_a_violation_nobody_has_cut_evidence_for_carries_no_keys(con):
    _source(con)
    _violation(con)

    (item,), _ = list_for_setup(con, "s1", None, None, limit=10, offset=0)

    # All three None together — the state that says nothing will ever build this,
    # which a reader has to tell apart from 'pending'.
    assert (item["thumbnail_key"], item["clip_key"], item["evidence_status"]) == (
        None,
        None,
        None,
    )


# --- reading one violation, blob and all -----------------------------------


def test_the_detail_read_carries_the_trajectories_the_list_leaves_behind(con):
    _source(con)
    violation_id = record(
        con,
        ViolationCreate(
            site_id="s1",
            source_id="src1",
            frame_index=912,
            type=ViolationType.RED_LIGHT_RUNNING,
            detected_at=DETECTED_AT,
            metadata=ViolationMetadata(
                vehicles=[
                    TrackSummary(
                        track_id=19,
                        trajectory=[(1.0, 2.0), None],
                        speed=[4.5, None],
                        frame_idxs=[911, 912],
                        bboxes=[(0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0)],
                    )
                ],
                violator_track_id=19,
            ),
        ),
    )

    violation = get_with_metadata(con, violation_id)

    assert violation["id"] == violation_id
    assert violation["metadata"]["violator_track_id"] == 19
    assert violation["metadata"]["vehicles"][0]["frame_idxs"] == [911, 912]
    # The same page read through the list carries none of that.
    page, _ = list_for_setup(con, "s1", None, None, limit=10, offset=0)
    assert "metadata" not in page[0]


def test_a_violation_that_does_not_exist_reads_as_nothing(con):
    assert get_with_metadata(con, "nope") is None


def test_a_violation_whose_blob_is_missing_is_still_a_violation(con):
    # `record` writes both rows in one transaction so this should not arise. If it
    # ever does, the row is the record of what happened — answering "no such
    # violation" would lose it.
    _source(con)
    violation_id = _violation(con)
    con.execute("DELETE FROM violation_metadata WHERE traffic_violation_id = ?", [violation_id])

    violation = get_with_metadata(con, violation_id)

    assert violation["id"] == violation_id
    assert violation["metadata"] is None


def test_a_violation_nobody_has_explained_carries_no_explanation(con):
    _source(con)
    violation_id = _violation(con)

    violation = get_with_metadata(con, violation_id)

    assert violation["status"] == ViolationStatus.DETECTED.value
    assert violation["explanation"] is None
    assert violation["severity"] is None
    assert violation["explanation_json"] is None


# --- writing an explanation ------------------------------------------------


def _explanation(**overrides):
    return ViolationExplanation(
        **{
            "explanation": "A vehicle drove into the junction after the signal had turned red.",
            "severity": Severity.MEDIUM,
            "severity_basis": ["other traffic was moving through at the time"],
            "evidence_strength": EvidenceStrength.WEAK,
            "evidence_basis": ["the record cannot confirm the signal was red"],
            "license_plate": LicensePlateAssessment(
                recoverability=PlateRecoverability.INCONCLUSIVE,
                reasoning="The vehicle stays distant and there is no plate recognition here.",
            ),
            "observations": ["Several objects counted as vehicles never move at all."],
            "evidence_concerns": ["Disregard any speed shown — the calibration is faulty."],
            "confidence": 0.6,
            **overrides,
        }
    )


def _explain(con, violation_id, explanation=None):
    explanation = explanation or _explanation()
    set_explanation(
        con,
        violation_id,
        explanation.explanation,
        explanation.severity.value,
        explanation.model_dump_json(),
    )
    return explanation


def test_an_explanation_lands_on_the_flat_columns_and_the_blob_together(con):
    _source(con)
    violation_id = _violation(con)

    _explain(con, violation_id)

    violation = get_with_metadata(con, violation_id)
    assert violation["explanation"] == (
        "A vehicle drove into the junction after the signal had turned red."
    )
    assert violation["severity"] == "MEDIUM"
    # The whole answer survives the round trip, including the field the flat columns
    # cannot hold — which is the reason the JSON column exists.
    stored = ViolationExplanation.model_validate_json(violation["explanation_json"])
    assert stored.evidence_concerns == [
        "Disregard any speed shown — the calibration is faulty."
    ]
    assert stored.evidence_strength is EvidenceStrength.WEAK
    assert stored.license_plate is not None
    assert stored.severity_basis == ["other traffic was moving through at the time"]


def test_explaining_a_violation_marks_it_explained(con):
    _source(con)
    violation_id = _violation(con)

    _explain(con, violation_id)

    assert get_with_metadata(con, violation_id)["status"] == ViolationStatus.EXPLAINED.value


def test_the_list_renders_a_severity_without_parsing_any_json(con):
    # The whole reason `explanation` and `severity` are duplicated onto their own
    # columns: a page of violations shows both without reaching for the blob.
    _source(con)
    violation_id = _violation(con)
    _explain(con, violation_id)

    page, _ = list_for_setup(con, "s1", None, None, limit=10, offset=0)

    assert page[0]["severity"] == "MEDIUM"
    assert page[0]["explanation"] == (
        "A vehicle drove into the junction after the signal had turned red."
    )


def test_explaining_a_violation_touches_its_updated_at(con):
    _source(con)
    violation_id = _violation(con)
    con.execute(
        "UPDATE traffic_violations SET updated_at = '2000-01-01 00:00:00' WHERE id = ?",
        [violation_id],
    )

    _explain(con, violation_id)

    updated_at = con.execute(
        "SELECT updated_at FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone()[0]
    assert not str(updated_at).startswith("2000")


def test_explaining_one_violation_leaves_the_others_alone(con):
    _source(con)
    explained = _violation(con)
    untouched = _violation(con)

    _explain(con, explained)

    other = get_with_metadata(con, untouched)
    assert other["explanation"] is None
    assert other["status"] == ViolationStatus.DETECTED.value


def test_a_second_explanation_replaces_the_first(con):
    # Nothing here refuses to overwrite. Deciding not to re-explain is the caller's,
    # where the cost of the call is understood.
    _source(con)
    violation_id = _violation(con)
    _explain(con, violation_id)

    _explain(con, violation_id, _explanation(explanation="Revised.", severity=Severity.HIGH))

    violation = get_with_metadata(con, violation_id)
    assert violation["explanation"] == "Revised."
    assert violation["severity"] == "HIGH"
