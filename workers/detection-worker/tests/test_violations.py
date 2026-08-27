import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.models.detection import ViolationType
from evidence_collector import TrackWindow
from shared.models.detection import DetectionJob, FrameRange, JobSource
from shared.models.violation import TrackSummary, ViolationCreate, ViolationMetadata
from violation_detector import Violation

from detection_worker import context
from detection_worker.violations import detected_at, record, summary, to_create

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


def test_record_writes_the_documents_the_violation_was_judged_against(con):
    """The reader needs both: to filter a site's violations down to the setup it runs
    under now, and to draw the evidence with the polygons that actually convicted."""
    for table, doc_id in (("camera_calibrations", "cal-1"), ("configurations", "cfg-1")):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version) VALUES (?, 's1', 'f1', 1)",
            [doc_id],
        )

    violation_id = record(
        con, _violation(calibration_id="cal-1", configuration_id="cfg-1")
    )

    assert con.execute(
        "SELECT calibration_id, configuration_id FROM traffic_violations WHERE id = ?",
        [violation_id],
    ).fetchone() == ("cal-1", "cfg-1")


def test_a_violation_from_an_uncalibrated_site_records_no_documents(con):
    """Not an edge case: detection runs on a site with a video and no camera model, so
    the write path has to accept one rather than fail on exactly those sites."""
    violation_id = record(con, _violation())

    assert con.execute(
        "SELECT calibration_id, configuration_id FROM traffic_violations WHERE id = ?",
        [violation_id],
    ).fetchone() == (None, None)


def test_a_violation_naming_a_source_that_does_not_exist_leaves_nothing_behind(con):
    """A row that cannot locate its own footage is a detection nobody can review, so
    it is refused rather than written and puzzled over later."""
    with pytest.raises(sqlite3.IntegrityError):
        record(con, _violation(source_id="no-such-source"))

    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM violation_metadata").fetchone()[0] == 0


def test_no_evidence_frames_are_recorded(con):
    """Still empty, and now for a different reason.

    The durable artifact this field was held open for exists — but it is a thumbnail
    and a clip, on the row, where the list endpoint can reach them without joining this
    table. Nothing writes `frames`.
    """
    violation_id = record(con, _violation())

    blob = con.execute(
        "SELECT json_blob FROM violation_metadata WHERE traffic_violation_id = ?",
        [violation_id],
    ).fetchone()[0]
    assert ViolationMetadata.model_validate_json(blob).frames == []


# --- assembling the row -------------------------------------------------------

ANCHOR = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)


def _context(**overrides) -> context.JobContext:
    """A resolved job context, carrying nothing but the anchor unless asked.

    The default is the uncalibrated site — no calibration, no configuration — because
    that is a normal state and the one a row still has to be writable from.
    """
    return context.JobContext(source_created_at=ANCHOR, **overrides)


def _job(fps: float | None = 30.0) -> DetectionJob:
    return DetectionJob(
        id="job-1",
        site_id="s1",
        source=JobSource(source_id="src-1", version=3, key="video/f1/a.mp4", fps=fps),
        frame_range=FrameRange(start=0, end=1000),
        types=[ViolationType.RED_LIGHT_RUNNING],
    )


def _scene(*windows: TrackWindow) -> dict[int, TrackWindow]:
    """What the analyzer hands over: every track the buffer held, keyed by id."""
    return {window.track_id: window for window in windows}


def _window(track_id: int = 7) -> TrackWindow:
    return TrackWindow(
        track_id=track_id,
        frame_indices=(898, 899, 900),
        positions=((1.0, 2.0), (1.5, 2.5), (2.0, 3.0)),
        speeds=(11.0, 12.0, 13.0),
        bboxes=((0, 0, 10, 10), (1, 1, 11, 11), (2, 2, 12, 12)),
        class_names=("car", "car", "car"),
        timestamps=(None, None, None),
    )


def test_a_window_becomes_a_summary_field_for_field():
    # A rename and nothing else. The evidence package answers in its own vocabulary
    # because it depends on nothing, and this is the whole cost of that.
    assert summary(_window()) == TrackSummary(
        track_id=7,
        trajectory=[(1.0, 2.0), (1.5, 2.5), (2.0, 3.0)],
        speed=[11.0, 12.0, 13.0],
        frame_idxs=[898, 899, 900],
        bboxes=[(0, 0, 10, 10), (1, 1, 11, 11), (2, 2, 12, 12)],
    )


def test_detected_at_is_the_anchor_plus_the_offset_into_the_footage():
    assert detected_at(ANCHOR, frame_index=900, fps=30.0) == ANCHOR + timedelta(seconds=30)


def test_a_source_with_no_frame_rate_cannot_place_a_violation_in_time():
    # Visibly wrong rather than quietly wrong: every violation in such a job lands on
    # the anchor, instead of on a moment computed from a frame rate nobody measured.
    assert detected_at(ANCHOR, frame_index=900, fps=None) == ANCHOR
    assert detected_at(ANCHOR, frame_index=900, fps=0) == ANCHOR


def test_a_violation_becomes_a_row_that_can_find_its_own_footage():
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window()),
        _context(),
    )

    assert created.source_id == "src-1"
    assert created.frame_index == 900
    assert created.type is ViolationType.RED_LIGHT_RUNNING
    assert created.detected_at == ANCHOR + timedelta(seconds=30)


def test_the_row_pins_what_the_violation_was_judged_against():
    # Not the versions the job named but the rows they resolved to, so the reader can
    # tell which violations hold under the site's current setup and can draw the
    # evidence with the polygons that actually convicted, not whatever is current now.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window()),
        _context(calibration_id="cal-1", configuration_id="cfg-1"),
    )

    assert created.calibration_id == "cal-1"
    assert created.configuration_id == "cfg-1"


def test_a_job_with_no_documents_pins_nothing_rather_than_guessing():
    # A site with a video and no calibration is ordinary, and None says exactly that.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window()),
        _context(),
    )

    assert created.calibration_id is None
    assert created.configuration_id is None


def test_the_row_carries_the_violation_s_own_frame_not_the_loop_s():
    # A module working on a clip reports several frames late, and recording the loop's
    # position would misdate it by the length of the window.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=870),
        _scene(_window()),
        _context(),
    )

    assert created.frame_index == 870


def test_the_violator_is_summarised_as_a_vehicle():
    created = to_create(
        _job(), Violation(type="red_light_running", track_id=7, frame_index=900), _scene(_window()), _context()
    )

    assert [v.track_id for v in created.metadata.vehicles] == [7]


def test_a_violation_with_no_window_still_becomes_a_row():
    # What the rules were holding at the end of a job. A violation with no history is
    # still a violation.
    created = to_create(
        _job(), Violation(type="red_light_running", track_id=7, frame_index=900), {}, _context()
    )

    assert created.metadata.vehicles == []


def _pedestrian(track_id: int = 12) -> TrackWindow:
    return TrackWindow(
        track_id=track_id,
        frame_indices=(898, 899, 900),
        positions=(None, None, None),
        speeds=(None, None, None),
        bboxes=((5, 5, 9, 15), (6, 5, 10, 15), (7, 5, 11, 15)),
        class_names=("person", "person", "person"),
        timestamps=(None, None, None),
    )


def test_the_counterparty_is_recorded_beside_the_vehicle():
    """The pedestrian rule's whole subject is somebody the module never names in what
    it returns. Their window is also the one thing that could not be recovered from the
    footage later without knowing which track to look for."""
    created = to_create(
        _job(),
        Violation(type="pedestrian_right_of_way", track_id=7, frame_index=900),
        _scene(_window(), _pedestrian()),
        _context(),
    )

    assert [p.track_id for p in created.metadata.pedestrians] == [12]
    assert [v.track_id for v in created.metadata.vehicles] == [7]


def test_the_convicted_track_stays_identifiable_in_a_crowd():
    # The whole reason violator_track_id exists: `vehicles[0]` used to be the answer
    # because there was never more than one, and now there is.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window(track_id=3), _window(), _window(track_id=9)),
        _context(),
    )

    assert created.metadata.violator_track_id == 7
    assert [v.track_id for v in created.metadata.vehicles] == [3, 7, 9]


def test_a_bystander_is_recorded_without_being_accused():
    """Who else was there is what a violation gets reviewed against. Which of them was
    in the crossing is a question about polygons, answered by whoever holds them."""
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window(), _window(track_id=9)),
        _context(),
    )

    assert [v.track_id for v in created.metadata.vehicles] == [7, 9]
    assert created.metadata.violator_track_id == 7


def test_an_empty_crossing_records_no_pedestrians():
    # Empty stays the ordinary answer; it just is not the only one any more.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(_window()),
        _context(),
    )

    assert created.metadata.pedestrians == []


def test_a_class_no_rule_has_a_word_for_is_kept_rather_than_dropped():
    # Wrong bucket, kept record. Dropping it would contradict recording the scene.
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(
            _window(),
            TrackWindow(
                track_id=21,
                frame_indices=(900,),
                positions=(None,),
                speeds=(None,),
                bboxes=((0, 0, 4, 8),),
                class_names=("traffic light",),
                timestamps=(None,),
            ),
        ),
        _context(),
    )

    assert [v.track_id for v in created.metadata.vehicles] == [7, 21]


def test_a_row_assembled_this_way_is_writable(con):
    """The two halves meet: what to_create builds is what record takes."""
    created = to_create(
        _job(), Violation(type="red_light_running", track_id=7, frame_index=900), _scene(_window()), _context()
    )

    violation_id = record(con, created)

    assert con.execute(
        "SELECT source_id, frame_index FROM traffic_violations WHERE id = ?", [violation_id]
    ).fetchone() == ("src-1", 900)


def test_an_unprojected_window_is_summarised_without_inventing_positions():
    """A job with no calibration writes a trajectory of Nones beside a full set of
    boxes. Honest about what was known, rather than plausible about what was not."""
    window = TrackWindow(
        track_id=7,
        frame_indices=(899, 900),
        positions=(None, None),
        speeds=(None, None),
        bboxes=((0, 0, 10, 10), (1, 1, 11, 11)),
        class_names=("car", "car"),
        timestamps=(None, None),
    )

    written = summary(window)

    assert written.trajectory == [None, None]
    assert written.speed == [None, None]
    assert written.bboxes == [(0, 0, 10, 10), (1, 1, 11, 11)]


def test_a_row_for_an_uncalibrated_job_survives_the_round_trip(con):
    """The failure this guards is a validation error at write time, on exactly the
    sites that have no calibration — which is a normal state, not an edge case."""
    created = to_create(
        _job(),
        Violation(type="red_light_running", track_id=7, frame_index=900),
        _scene(
            TrackWindow(
                track_id=7,
                frame_indices=(900,),
                positions=(None,),
                speeds=(None,),
                bboxes=((0, 0, 10, 10),),
                class_names=("car",),
                timestamps=(None,),
            )
        ),
        _context(),
    )

    violation_id = record(con, created)

    blob = con.execute(
        "SELECT json_blob FROM violation_metadata WHERE traffic_violation_id = ?",
        [violation_id],
    ).fetchone()[0]
    assert ViolationMetadata.model_validate_json(blob).vehicles[0].trajectory == [None]
