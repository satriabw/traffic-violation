from datetime import datetime, timezone

import pytest

from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import EvidenceTarget, record
from shared.models.detection import ViolationType
from shared.models.evidence import EvidenceJob
from shared.models.violation import ViolationCreate
from shared.queue.memory import InMemoryQueue

from evidence_worker import worker
from evidence_worker.cut import CutFailed

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
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id, metadata)"
        " VALUES ('src1', 's1', 1, 'video', 'f1', '{\"fps\": 10.0}')"
    )
    return con


def _violation(con, frame_index=100):
    return record(
        con,
        ViolationCreate(
            site_id="s1",
            source_id="src1",
            frame_index=frame_index,
            type=ViolationType.RED_LIGHT_RUNNING,
            detected_at=DETECTED_AT,
        ),
    )


class Cuts:
    """Stand-ins for the two ffmpeg calls, recording what they were asked for."""

    def __init__(self, fail_on=None):
        self.thumbnails: list[tuple[str, float]] = []
        self.clips: list[tuple[str, float, float]] = []
        self._fail_on = fail_on

    def thumbnail(self, url, seconds, path):
        if self._fail_on == "thumbnail":
            raise CutFailed("no frame there")
        self.thumbnails.append((url, seconds))
        open(path, "wb").write(b"jpeg")

    def clip(self, url, start, duration, path):
        if self._fail_on == "clip":
            raise CutFailed("nothing to copy")
        self.clips.append((url, start, duration))
        open(path, "wb").write(b"mp4")


class Uploads:
    def __init__(self, error=None):
        self.put: list[tuple[str, str]] = []
        self._error = error

    def __call__(self, key, path, content_type=None):
        if self._error:
            raise self._error
        self.put.append((key, content_type))
        return key


def _handler(con, cuts=None, uploads=None):
    cuts = cuts or Cuts()
    uploads = uploads or Uploads()
    handle = worker.make_handler(
        con,
        sign=lambda key: f"https://signed/{key}",
        put=uploads,
        cut_thumbnail=cuts.thumbnail,
        cut_clip=cuts.clip,
    )
    return handle, cuts, uploads


def _row(con, violation_id):
    return con.execute(
        "SELECT evidence_status, thumbnail_key, clip_key FROM traffic_violations"
        " WHERE id = ?",
        [violation_id],
    ).fetchone()


# --- the window ---------------------------------------------------------------


def test_the_clip_is_the_lead_up_the_record_was_kept_over():
    # The same number the detector sized its ring buffer with, so the blob holds boxes
    # for exactly these frames. Anything else and a reviewer drawing them over the clip
    # watches them start late or run out early.
    at, start, duration = worker.window(
        EvidenceTarget(key="k", frame_index=100, fps=10.0), evidence_seconds=5.0
    )

    assert at == 10.0
    assert start == pytest.approx(5.0)
    assert duration == pytest.approx(5.1)


def test_the_clip_ends_on_the_frame_the_rule_fired_on_rather_than_before_it():
    # FrameBuffer.over's `+ 1`, in seconds. The frame a rule fires on is part of its
    # window, so the clip has to reach the far side of that frame — stopping at its
    # leading edge would end the evidence one frame before the violation.
    at, start, duration = worker.window(
        EvidenceTarget(key="k", frame_index=100, fps=10.0), evidence_seconds=5.0
    )

    assert start + duration == pytest.approx(at + 1 / 10.0)


def test_a_site_that_asks_for_more_approach_gets_a_longer_clip():
    # A junction with a long sight line wants more; a tight one-way needs less. That is
    # a site's decision, and it is the only place the length is decided.
    _, short_start, _ = worker.window(
        EvidenceTarget(key="k", frame_index=300, fps=10.0), evidence_seconds=2.0
    )
    _, long_start, _ = worker.window(
        EvidenceTarget(key="k", frame_index=300, fps=10.0), evidence_seconds=12.0
    )

    assert short_start == pytest.approx(28.0)
    assert long_start == pytest.approx(18.0)


def test_a_violation_near_the_start_of_a_video_does_not_seek_backwards():
    # A negative -ss is not a shorter clip, it is an ffmpeg invocation that fails.
    at, start, duration = worker.window(
        EvidenceTarget(key="k", frame_index=5, fps=10.0), evidence_seconds=5.0
    )

    assert start == 0.0
    # Short rather than wrong — the same truncation the record itself carries when a
    # violation reaches back past the start of its own chunk.
    assert start + duration == pytest.approx(at + 1 / 10.0)


# --- the happy path -----------------------------------------------------------


def test_a_cut_violation_ends_up_ready_with_both_keys(con):
    violation_id = _violation(con)
    handle, _, _ = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=violation_id))

    assert _row(con, violation_id) == (
        "ready",
        f"evidence/{violation_id}/thumbnail.jpg",
        f"evidence/{violation_id}/clip.mp4",
    )


def test_both_objects_are_namespaced_under_the_violation_they_belong_to(con):
    # Two violations from one job would otherwise write the same two keys, and the
    # second would silently overwrite the first's evidence.
    first, second = _violation(con, 100), _violation(con, 200)
    handle, _, uploads = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=first))
    handle(EvidenceJob(evidence_seconds=5.0, violation_id=second))

    assert [key for key, _ in uploads.put] == [
        f"evidence/{first}/thumbnail.jpg",
        f"evidence/{first}/clip.mp4",
        f"evidence/{second}/thumbnail.jpg",
        f"evidence/{second}/clip.mp4",
    ]


def test_each_object_is_stored_as_what_it_is(con):
    # A browser will not play an mp4 served as application/octet-stream.
    handle, _, uploads = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=_violation(con)))

    assert [content_type for _, content_type in uploads.put] == ["image/jpeg", "video/mp4"]


def test_the_cut_is_made_against_a_signed_url_for_the_pinned_source(con):
    handle, cuts, _ = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=_violation(con, frame_index=100)))

    assert cuts.thumbnails == [("https://signed/video/f1/a.mp4", 10.0)]
    assert cuts.clips[0][0] == "https://signed/video/f1/a.mp4"


# --- what fails, and how far it gets ------------------------------------------


@pytest.mark.parametrize("stage", ["thumbnail", "clip"])
def test_a_cut_that_fails_marks_the_violation_and_uploads_nothing(con, stage):
    violation_id = _violation(con)
    handle, _, uploads = _handler(con, cuts=Cuts(fail_on=stage))

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=violation_id))

    assert _row(con, violation_id) == ("failed", None, None)
    # Nothing half-done in storage: the row never names a key, and no key exists for a
    # row to name.
    assert uploads.put == []


def test_a_violation_whose_source_has_no_frame_rate_is_not_guessed_at(con):
    # A frame index is only a position in time alongside a rate. Assuming 25 or 30 seeks
    # to the wrong second of a real video, which is evidence of something that did not
    # happen — worse than no evidence.
    con.execute("UPDATE site_sources SET metadata = '{}' WHERE id = 'src1'")
    violation_id = _violation(con)
    handle, cuts, _ = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id=violation_id))

    assert _row(con, violation_id) == ("failed", None, None)
    assert cuts.thumbnails == []


def test_a_job_for_a_violation_that_is_not_there_changes_nothing(con):
    existing = _violation(con)
    handle, cuts, _ = _handler(con)

    handle(EvidenceJob(evidence_seconds=5.0, violation_id="no-such-violation"))

    assert cuts.thumbnails == []
    # The failing write matches no row, which is the right amount of nothing to do.
    assert _row(con, existing) == (None, None, None)


def test_a_failed_cut_does_not_stop_the_worker(con):
    # Where this diverges from detection-worker, whose raising handler stops the
    # process. Here the failure has somewhere to go — the row — so the next violation
    # still gets its turn.
    doomed, fine = _violation(con, 100), _violation(con, 200)
    cuts = Cuts()
    handle = worker.make_handler(
        con,
        sign=lambda key: key,
        put=Uploads(),
        cut_thumbnail=lambda url, seconds, path: (_ for _ in ()).throw(CutFailed("no"))
        if seconds == 10.0
        else cuts.thumbnail(url, seconds, path),
        cut_clip=cuts.clip,
    )
    queue = InMemoryQueue()
    queue.enqueue(EvidenceJob(evidence_seconds=5.0, violation_id=doomed))
    queue.enqueue(EvidenceJob(evidence_seconds=5.0, violation_id=fine))

    assert worker.run(queue, handle) == 2
    assert _row(con, doomed)[0] == "failed"
    assert _row(con, fine)[0] == "ready"


def test_a_bucket_that_refuses_a_write_stops_the_worker_rather_than_the_violation(con):
    # Not a fact about this violation. Marking it failed would quietly do the same to
    # every job behind it, and leave nothing saying the credentials were wrong.
    violation_id = _violation(con)
    handle, _, _ = _handler(con, uploads=Uploads(error=PermissionError("AccessDenied")))

    with pytest.raises(PermissionError):
        handle(EvidenceJob(evidence_seconds=5.0, violation_id=violation_id))

    assert _row(con, violation_id) == (None, None, None)


# --- the loop -----------------------------------------------------------------


def test_the_worker_drains_the_queue(con):
    handle, _, _ = _handler(con)
    queue = InMemoryQueue()
    ids = [_violation(con, frame_index=i * 100) for i in range(1, 4)]
    for violation_id in ids:
        queue.enqueue(EvidenceJob(evidence_seconds=5.0, violation_id=violation_id))

    assert worker.run(queue, handle) == 3
    assert [_row(con, violation_id)[0] for violation_id in ids] == ["ready"] * 3


def test_the_worker_stops_after_max_jobs(con):
    handle, _, _ = _handler(con)
    queue = InMemoryQueue()
    for i in range(1, 4):
        queue.enqueue(EvidenceJob(evidence_seconds=5.0, violation_id=_violation(con, frame_index=i * 100)))

    assert worker.run(queue, handle, max_jobs=1) == 1
