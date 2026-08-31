import sqlite3

import numpy as np
import pytest
import supervision as sv
from trajectory_collector import Trajectory
from evidence_collector import TrackWindow
from violation_detector import Violation
from shared.models.detection import DetectionJob, FrameRange, JobSource, ViolationType
from shared.queue.memory import InMemoryQueue

from detection_worker.analysis.frame_result import FrameResult
from detection_worker import context
from detection_worker.context import JobContext
from detection_worker.video.reader import VideoUnavailable
from detection_worker.worker import make_handler, run


def _job(job_id: str, key: str = "video/file-1/clip.mp4", start: int = 0, end: int = 100) -> DetectionJob:
    return DetectionJob(
        id=job_id,
        site_id="site-1",
        source=JobSource(source_id="source-1", version=3, key=key, fps=30.0, total_frames=end),
        frame_range=FrameRange(start=start, end=end),
        types=[ViolationType.RED_LIGHT_RUNNING],
    )


def _queue_of(*job_ids: str) -> InMemoryQueue:
    queue = InMemoryQueue()
    for job_id in job_ids:
        queue.enqueue(_job(job_id))
    return queue


def _reader_of(frame_count: int):
    """A stand-in for read_frames that records what it was asked for."""
    calls: list[tuple[str, FrameRange]] = []

    def read(url, frame_range):
        calls.append((url, frame_range))
        for index in range(frame_range.start, frame_range.start + frame_count):
            yield index, np.zeros((2, 2, 3), dtype=np.uint8)

    read.calls = calls
    return read


class FakeAnalyzer:
    """A per-frame pipeline with nothing behind it.

    The handler's job is orchestration — sign, read, aggregate, log — so everything
    here can be exercised without a model, a tracker or a weights file. What detection
    and tracking do with a frame is the analyzer suite's business.
    """

    def __init__(
        self,
        job,
        job_context,
        detections_per_frame: int = 0,
        locates: bool = False,
        violates: bool = False,
        held: int = 0,
        window: int = 3,
        capacity: int = 3,
        bystanders: int = 0,
    ):
        self.job = job
        self.job_context = job_context
        self.analyzed: list[tuple[np.ndarray, int]] = []
        self.finished = 0
        self._count = detections_per_frame
        self._locates = locates
        self._violates = violates
        self._held = held
        # How long a window this analyzer hands back, against how long a one it could.
        # The handler's whole interest in evidence is telling those two apart.
        self._window = window
        self._capacity = capacity
        # Other tracks the ring held when the rule fired. The record keeps the whole
        # scene, so these arrive alongside the violator rather than being filtered out.
        self._bystanders = bystanders

    @property
    def evidence_capacity(self) -> int:
        """How long a window this job could hold. A property, as the real one is."""
        return self._capacity

    def finish(self) -> list[Violation]:
        """What the rules were still holding. Empty for every rule that ships today."""
        self.finished += 1
        return [
            Violation(type="red_light_running", track_id=track_id, frame_index=0)
            for track_id in range(1, self._held + 1)
        ]

    def _window_of(self, track_id: int, index: int, length: int) -> TrackWindow:
        return TrackWindow(
            track_id=track_id,
            frame_indices=tuple(range(index - length + 1, index + 1)),
            positions=((0.0, 0.0),) * length,
            speeds=(1.0,) * length,
            bboxes=((0.0, 0.0, 1.0, 1.0),) * length,
            class_names=("car",) * length,
            timestamps=(None,) * length,
        )

    def _scene(self, index: int) -> dict[int, TrackWindow]:
        """Track 1 is the one convicted; the bystanders are everyone else in the ring.

        Theirs are deliberately one frame long, so a test can tell a window the handler
        should count as short from ones it should ignore — with the whole scene in the
        record, an object that had only just walked into view is short every time and
        says nothing about whether the chunk overlap is big enough.
        """
        scene = {1: self._window_of(1, index, self._window)}
        for track_id in range(2, self._bystanders + 2):
            scene[track_id] = self._window_of(track_id, index, 1)
        return scene

    def analyze(self, frame: np.ndarray, index: int) -> FrameResult:
        self.analyzed.append((frame, index))
        if self._count == 0:
            return FrameResult(index=index, detections=sv.Detections.empty(), trajectories={})
        return FrameResult(
            index=index,
            # Whether a rule fires is the detector's business; the handler only counts
            # what it is handed.
            violations=[
                Violation(type="red_light_running", track_id=1, frame_index=index)
            ]
            if self._violates
            else [],
            evidence=self._scene(index) if self._violates else {},
            # Whether a job has trajectories at all depends on its calibration, which
            # the analyzer resolves — the handler only counts and logs them.
            trajectories={
                track_id: Trajectory(position=(0.0, 0.0), speed=1.0)
                for track_id in range(1, self._count + 1)
            }
            if self._locates
            else {},
            detections=sv.Detections(
                xyxy=np.array(
                    [[i, i, i + 10, i + 10] for i in range(self._count)], dtype=np.float32
                ),
                confidence=np.full(self._count, 0.9, dtype=np.float32),
                class_id=np.full(self._count, 2, dtype=np.int16),
                # The tracker numbers them 1..n every frame, so a multi-frame job sees
                # the same handful of ids over and over — which is what makes the
                # distinct-id count in the summary worth asserting on.
                tracker_id=np.arange(1, self._count + 1),
            ),
        )


def _analyzers(
    detections_per_frame: int = 0,
    locates: bool = False,
    violates: bool = False,
    held: int = 0,
    window: int = 3,
    capacity: int = 3,
    bystanders: int = 0,
):
    """An analyzer factory that records every analyzer it was asked to build."""
    made: list[FakeAnalyzer] = []

    def factory(job, job_context):
        analyzer = FakeAnalyzer(
            job, job_context, detections_per_frame, locates, violates, held, window,
            capacity, bystanders,
        )
        made.append(analyzer)
        return analyzer

    factory.made = made
    return factory


class FakeStore:
    """Records every violation it was asked to write, and numbers them."""

    def __init__(self, fails_on: int | None = None):
        self.saved: list = []
        self._fails_on = fails_on

    def __call__(self, violation) -> str:
        self.saved.append(violation)
        if self._fails_on is not None and len(self.saved) == self._fails_on:
            raise sqlite3.IntegrityError("no such site")
        return f"v-{len(self.saved)}"


class FakeSourceStatus:
    """Records the run's status transitions instead of writing them.

    A list rather than a last-value, because the order is the thing being asserted: a
    source that ends 'completed' having never said 'processing' would look identical
    otherwise, and that is exactly the bug worth catching.
    """

    def __init__(self):
        self.transitions: list[str] = []

    def __call__(self, source_id: str, status) -> None:
        self.transitions.append(status.value)


def _handler(
    sign=None,
    read=None,
    new_analyzer=None,
    load_context=None,
    save=None,
    queue_evidence=None,
    set_status=None,
):
    return make_handler(
        # A site with neither document is the default here: these tests are about
        # frames, detections and ids, and context has its own suite.
        load_context or (lambda job: JobContext()),
        new_analyzer or _analyzers(),
        save or FakeStore(),
        queue_evidence=queue_evidence or (lambda violation_id, seconds: None),
        set_status=set_status or (lambda source_id, status: None),
        sign=sign or (lambda key: "u"),
        read=read if read is not None else _reader_of(frame_count=0),
    )


def test_run_hands_every_queued_job_to_the_handler_in_order():
    handled: list[DetectionJob] = []

    run(_queue_of("job-0", "job-1", "job-2"), handle=handled.append)

    assert [job.id for job in handled] == ["job-0", "job-1", "job-2"]


def test_run_returns_the_number_of_jobs_handled():
    assert run(_queue_of("job-0", "job-1"), handle=lambda job: None) == 2


def test_run_stops_on_an_empty_queue():
    # Against Redis, consume() blocks instead of returning None, so this same loop
    # runs forever there — the queue decides, not the worker.
    assert run(InMemoryQueue(), handle=lambda job: None) == 0


def test_max_jobs_stops_the_loop_and_leaves_the_rest_queued():
    queue = _queue_of("job-0", "job-1", "job-2")

    assert run(queue, handle=lambda job: None, max_jobs=2) == 2
    assert queue.consume().id == "job-2"


def test_the_handler_signs_the_key_the_job_carried():
    # The url is minted here, not at enqueue time: a signature that expired in a
    # backlog would fail after the worker had already started.
    signed: list[str] = []
    read = _reader_of(frame_count=0)

    _handler(sign=lambda key: signed.append(key) or f"https://r2/{key}?sig", read=read)(
        _job("job-0", key="video/file-9/clip.mp4")
    )

    assert signed == ["video/file-9/clip.mp4"]


def test_the_handler_reads_the_signed_url_over_the_job_s_frame_range():
    read = _reader_of(frame_count=0)

    _handler(sign=lambda key: "https://r2/signed", read=read)(
        _job("job-0", start=900, end=1800)
    )

    url, frame_range = read.calls[0]
    assert url == "https://r2/signed"
    assert (frame_range.start, frame_range.end) == (900, 1800)


def test_the_handler_reads_every_frame_and_logs_the_count(caplog):
    read = _reader_of(frame_count=4)

    with caplog.at_level("INFO"):
        _handler(read=read)(_job("job-0"))

    # The count is the evidence every requested frame actually decoded, so it has to
    # reach the log.
    assert "read=4" in caplog.text
    assert "source=source-1 v3" in caplog.text


def test_a_video_that_cannot_be_opened_stops_the_worker(caplog):
    # No retries and no dead-letter queue yet, so a failing job must fail loudly
    # rather than be dropped.
    def read(url, frame_range):
        raise VideoUnavailable("expired url")
        yield  # pragma: no cover — makes this a generator, as read_frames is

    queue = _queue_of("job-0", "job-1")

    with pytest.raises(VideoUnavailable):
        run(queue, handle=_handler(read=read))

    # The second job is still queued: nothing was silently consumed on the way out.
    assert queue.consume().id == "job-1"


# --- the analyzer's lifetime --------------------------------------------------


def test_every_frame_read_reaches_the_analyzer_with_its_index():
    # Absolute indices, straight from the reader: a violation is recorded against the
    # frame's position in the video, not its position in this run.
    analyzers = _analyzers()

    _handler(read=_reader_of(frame_count=3), new_analyzer=analyzers)(
        _job("job-0", start=900, end=1800)
    )

    assert [index for _, index in analyzers.made[0].analyzed] == [900, 901, 902]


def test_the_analyzer_is_built_once_per_job_not_once_per_frame():
    # The lifetime this whole design turns on. An analyzer built per frame would carry
    # a tracker with no memory of the previous one, so nothing would ever hold an id
    # for two frames.
    analyzers = _analyzers()

    _handler(read=_reader_of(frame_count=6), new_analyzer=analyzers)(_job("job-0"))

    assert len(analyzers.made) == 1


def test_each_job_gets_its_own_analyzer():
    # The other half of it: state from one job must not survive into the next, or a
    # track from one site could be re-matched against another's.
    analyzers = _analyzers()
    handle = _handler(read=_reader_of(frame_count=2), new_analyzer=analyzers)

    handle(_job("job-0"))
    handle(_job("job-1"))

    assert len(analyzers.made) == 2
    assert analyzers.made[0] is not analyzers.made[1]


def test_the_analyzer_is_built_from_the_job():
    # It is what carries the frame rate the tracker and the trajectory collector are
    # both scaled by.
    analyzers = _analyzers()

    _handler(read=_reader_of(frame_count=1), new_analyzer=analyzers)(_job("job-0"))

    assert analyzers.made[0].job.id == "job-0"
    assert analyzers.made[0].job.source.fps == 30.0


def test_the_analyzer_is_built_from_the_context_the_job_was_pinned_to():
    # The calibration the trajectory collector projects with. It has to be the one the
    # job named, not whatever is active now — which is the whole reason the handler
    # resolves context before it builds the analyzer.
    analyzers = _analyzers()
    pinned = JobContext(calibration={"camera_matrix": "v3"})

    _handler(
        read=_reader_of(frame_count=1),
        new_analyzer=analyzers,
        load_context=lambda job: pinned,
    )(_job("job-0"))

    assert analyzers.made[0].job_context is pinned


def test_the_analyzer_is_built_after_the_context_is_resolved():
    # A job naming a calibration that is not there should fail before anything spends
    # minutes decoding — and before a collector is built from a document that does not
    # exist.
    built: list[str] = []

    def load_context(job):
        built.append("context")
        raise context.ContextMissing("calibration v3 is not there")

    def new_analyzer(job, job_context):
        built.append("analyzer")  # pragma: no cover — must never be reached

    with pytest.raises(context.ContextMissing):
        _handler(load_context=load_context, new_analyzer=new_analyzer)(_job("job-0"))

    assert built == ["context"]


# --- the summary --------------------------------------------------------------


def test_the_summary_logs_how_much_was_detected_and_tracked(caplog):
    with caplog.at_level("INFO"):
        _handler(read=_reader_of(frame_count=3), new_analyzer=_analyzers(2))(_job("job-0"))

    # 3 frames x 2 detections, and the fake analyzer numbers them 1..2 every frame, so
    # two distinct ids across the job.
    assert "read=3 detections=6 tracks=2" in caplog.text


def test_the_summary_logs_how_many_tracks_were_put_on_the_ground(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3), new_analyzer=_analyzers(2, locates=True)
        )(_job("job-0"))

    assert "tracks=2 located=2" in caplog.text


def test_a_job_with_no_calibration_locates_nothing(caplog):
    # Normal, not a failure: without a calibration there is no ground plane. The count
    # is what makes the difference visible in a log rather than only in the database.
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3), new_analyzer=_analyzers(2, locates=False)
        )(_job("job-0"))

    assert "tracks=2 located=0" in caplog.text


def test_a_job_that_detects_nothing_still_logs_a_summary(caplog):
    with caplog.at_level("INFO"):
        _handler(read=_reader_of(frame_count=3), new_analyzer=_analyzers(0))(_job("job-0"))

    assert "read=3 detections=0 tracks=0" in caplog.text


def test_per_frame_detail_is_logged_at_debug_not_info(caplog):
    # A 30-second chunk is ~900 frames. Per-frame lines at INFO would bury the one
    # line anyone watching a normal run actually wants.
    with caplog.at_level("INFO"):
        _handler(read=_reader_of(frame_count=2), new_analyzer=_analyzers(1))(_job("job-0"))
    assert "frame 0" not in caplog.text

    caplog.clear()
    with caplog.at_level("DEBUG"):
        _handler(read=_reader_of(frame_count=2), new_analyzer=_analyzers(1))(_job("job-0"))
    assert "frame 0 detections=1 ids=[1]" in caplog.text


# --- violations ---------------------------------------------------------------


def test_the_summary_reports_what_the_rules_found(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3),
            new_analyzer=_analyzers(detections_per_frame=2, violates=True),
        )(_job("job-1"))

    # One per frame, from three frames.
    assert "violations=3" in caplog.text


def test_a_job_whose_rules_never_fire_reports_none(caplog):
    # Indistinguishable from a site with no configuration by this number alone, which
    # is why the configuration version is on the same line.
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3), new_analyzer=_analyzers(detections_per_frame=2)
        )(_job("job-1"))

    assert "violations=0" in caplog.text


def test_the_rules_are_drained_once_when_the_frames_run_out():
    # Once, after the loop — not per frame. A module working on a clip is holding a
    # partial one here, and draining it mid-run would cut every window short.
    analyzers = _analyzers()

    _handler(read=_reader_of(frame_count=3), new_analyzer=analyzers)(_job("job-1"))

    assert analyzers.made[0].finished == 1


def test_violations_held_back_until_the_end_still_reach_the_summary(caplog):
    # Without the drain these would be dropped in silence, and the last seconds of
    # every chunk would go unjudged.
    with caplog.at_level("INFO"):
        _handler(read=_reader_of(frame_count=2), new_analyzer=_analyzers(held=2))(_job("job-1"))

    assert "violations=2" in caplog.text


def test_a_job_with_no_frames_still_drains_the_rules():
    analyzers = _analyzers()

    _handler(read=_reader_of(frame_count=0), new_analyzer=analyzers)(_job("job-1"))

    assert analyzers.made[0].finished == 1


# --- the record that comes with a violation ----------------------------------


def test_every_violation_that_carried_a_window_is_counted(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3),
            new_analyzer=_analyzers(detections_per_frame=2, violates=True),
        )(_job("job-1"))

    assert "evidence=3" in caplog.text


def test_a_job_where_nothing_fired_carried_no_records(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3), new_analyzer=_analyzers(detections_per_frame=2)
        )(_job("job-1"))

    assert "evidence=0" in caplog.text
    assert "short=0" in caplog.text


def test_a_full_window_is_not_counted_as_a_short_one(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=2),
            new_analyzer=_analyzers(detections_per_frame=1, violates=True, window=3, capacity=3),
        )(_job("job-1"))

    assert "evidence=2" in caplog.text
    assert "short=0" in caplog.text


def test_a_window_cut_short_is_counted_separately(caplog):
    # The one that matters. A handful is ordinary — objects that had only just
    # appeared. Most of them, on a job that is not the first chunk of its video, means
    # the window is longer than the overlap between chunks and every record is missing
    # its approach.
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=2),
            new_analyzer=_analyzers(detections_per_frame=1, violates=True, window=2, capacity=151),
        )(_job("job-1"))

    assert "evidence=2" in caplog.text
    assert "short=2" in caplog.text


def test_a_bystander_with_a_short_window_is_not_counted_as_a_short_one(caplog):
    # The record keeps the whole scene, so most frames that fire carry objects that had
    # only just walked into view. Counting theirs would put `short=` at nearly the
    # number of tracks on every busy junction — burying the one thing it is for, which
    # is a violation reaching back past the start of its own chunk.
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=2),
            new_analyzer=_analyzers(
                detections_per_frame=1, violates=True, window=3, capacity=3, bystanders=4
            ),
        )(_job("job-1"))

    # Five windows recorded per firing frame, over two frames — the whole scene.
    assert "evidence=10" in caplog.text
    # None of them the violator's, whose window is full.
    assert "short=0" in caplog.text


def test_the_violator_s_own_short_window_is_still_counted(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=2),
            new_analyzer=_analyzers(
                detections_per_frame=1, violates=True, window=2, capacity=151, bystanders=4
            ),
        )(_job("job-1"))

    assert "short=2" in caplog.text


def test_the_whole_scene_reaches_the_record_not_only_the_violator(caplog):
    """Who else was there is what a violation gets reviewed against, and the reader
    holds the polygons that say which of them was in the crossing."""
    store = FakeStore()

    _handler(
        read=_reader_of(frame_count=1),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True, bystanders=2),
        save=store,
    )(_job("job-1"))

    [written] = store.saved
    assert [v.track_id for v in written.metadata.vehicles] == [1, 2, 3]
    assert written.metadata.violator_track_id == 1


def test_the_capacity_is_read_once_per_job_not_once_per_frame():
    # It is a property of the job. Asking per frame would be the same answer every
    # time, on every frame of a multi-minute chunk.
    class CountingAnalyzer(FakeAnalyzer):
        reads = 0

        @property
        def evidence_capacity(self):
            type(self).reads += 1
            return self._capacity

    def factory(job, job_context):
        return CountingAnalyzer(job, job_context, detections_per_frame=1, violates=True)

    _handler(read=_reader_of(frame_count=5), new_analyzer=factory)(_job("job-1"))

    assert CountingAnalyzer.reads == 1


def test_violations_held_back_until_the_end_do_not_invent_a_record(caplog):
    # They report on a frame the ring has already rolled past, so any window handed to
    # them would describe the end of the job rather than the moment convicted.
    with caplog.at_level("INFO"):
        _handler(read=_reader_of(frame_count=2), new_analyzer=_analyzers(held=2))(_job("job-1"))

    assert "violations=2" in caplog.text
    assert "evidence=0" in caplog.text


# --- writing the row ----------------------------------------------------------


def test_every_violation_becomes_a_row():
    store = FakeStore()

    _handler(
        read=_reader_of(frame_count=3),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True),
        save=store,
    )(_job("job-1"))

    assert len(store.saved) == 3


def test_a_job_where_nothing_fired_writes_nothing():
    store = FakeStore()

    _handler(read=_reader_of(frame_count=3), new_analyzer=_analyzers(1), save=store)(
        _job("job-1")
    )

    assert store.saved == []


def test_a_row_pins_the_video_it_came_from_and_the_frame():
    # The whole reason no frames are uploaded: without these the evidence can never be
    # re-derived. The frame is the violation's own, not the loop's position.
    store = FakeStore()

    _handler(
        read=_reader_of(frame_count=1),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True),
        save=store,
    )(_job("job-1", start=900, end=1000))

    written = store.saved[0]
    assert (written.site_id, written.source_id, written.frame_index) == (
        "site-1",
        "source-1",
        900,
    )


def test_a_row_carries_the_window_that_led_up_to_it():
    store = FakeStore()

    _handler(
        read=_reader_of(frame_count=1),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True, window=3),
        save=store,
    )(_job("job-1", start=900, end=1000))

    (vehicle,) = store.saved[0].metadata.vehicles
    assert vehicle.track_id == 1
    assert vehicle.frame_idxs == [898, 899, 900]


def test_no_evidence_frames_are_written():
    # Still empty, and now for a different reason: the thumbnail and the clip are
    # columns on the row, cut by evidence-worker, not entries in this list.
    store = FakeStore()

    _handler(
        read=_reader_of(frame_count=1),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True),
        save=store,
    )(_job("job-1"))

    assert store.saved[0].metadata.frames == []


def test_violations_held_back_until_the_end_are_still_written():
    # They come with no window — the ring rolled past the frame they report on — and a
    # violation with no history is still a violation.
    store = FakeStore()

    _handler(read=_reader_of(frame_count=2), new_analyzer=_analyzers(held=2), save=store)(
        _job("job-1")
    )

    assert len(store.saved) == 2
    assert store.saved[0].metadata.vehicles == []


def test_the_rows_written_are_counted_in_the_summary(caplog):
    with caplog.at_level("INFO"):
        _handler(
            read=_reader_of(frame_count=3),
            new_analyzer=_analyzers(detections_per_frame=1, violates=True),
            save=FakeStore(),
        )(_job("job-1"))

    assert "violations=3" in caplog.text
    assert "recorded=3" in caplog.text


def test_a_write_that_fails_stops_the_worker_and_keeps_what_landed():
    # Failing loudly beats losing work quietly, and each write is its own transaction —
    # so the rows already committed stay committed. Retries and a dead-letter queue are
    # marked FUTURE in the LLD.
    store = FakeStore(fails_on=2)

    with pytest.raises(sqlite3.IntegrityError):
        _handler(
            read=_reader_of(frame_count=4),
            new_analyzer=_analyzers(detections_per_frame=1, violates=True),
            save=store,
        )(_job("job-1"))

    assert len(store.saved) == 2


# --- asking for the evidence --------------------------------------------------


def test_every_row_written_is_queued_for_a_cut():
    store = FakeStore()
    queued: list[str] = []

    _handler(
        read=_reader_of(frame_count=3),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True),
        save=store,
        queue_evidence=lambda violation_id, seconds: queued.append(violation_id),
    )(_job("job-1"))

    # By the id `save` handed back, which is the only thing that identifies the row it
    # just wrote — the worker never sees a violation id before that.
    assert queued == ["v-1", "v-2", "v-3"]


def test_the_cut_is_asked_for_over_the_window_the_record_was_kept_over():
    # The same number the analyzer sized its ring buffer with, so the clip covers
    # exactly the frames the record holds boxes for. Anything else and the two disagree.
    asked: list[tuple[str, float]] = []

    _handler(
        load_context=lambda job: JobContext(evidence_seconds=12.0),
        read=_reader_of(frame_count=1),
        new_analyzer=_analyzers(detections_per_frame=1, violates=True),
        save=FakeStore(),
        queue_evidence=lambda violation_id, seconds: asked.append((violation_id, seconds)),
    )(_job("job-1"))

    assert asked == [("v-1", 12.0)]


def test_nothing_is_queued_for_a_violation_that_was_not_written():
    # The order the handler writes in. A job naming a violation that is not in the
    # database is one evidence-worker picks up, fails to find, and marks failed — a row
    # poisoned by the sequence it was written in rather than by anything about it.
    queued: list[str] = []

    with pytest.raises(sqlite3.IntegrityError):
        _handler(
            read=_reader_of(frame_count=3),
            new_analyzer=_analyzers(detections_per_frame=1, violates=True),
            save=FakeStore(fails_on=2),
            queue_evidence=lambda violation_id, seconds: queued.append(violation_id),
        )(_job("job-1"))

    assert queued == ["v-1"]


def test_violations_held_back_until_the_end_are_queued_too():
    # They carry no window, but the clip is cut from the source rather than from what
    # the ring held — so a drained violation has exactly as much evidence available as
    # one that fired mid-chunk, and skipping it would lose nothing but the evidence.
    queued: list[str] = []

    _handler(
        read=_reader_of(frame_count=2),
        new_analyzer=_analyzers(held=2),
        save=FakeStore(),
        queue_evidence=lambda violation_id, seconds: queued.append(violation_id),
    )(_job("job-1"))

    assert queued == ["v-1", "v-2"]


def test_a_job_where_nothing_fired_queues_nothing():
    queued: list[str] = []

    _handler(
        read=_reader_of(frame_count=3),
        new_analyzer=_analyzers(1),
        save=FakeStore(),
        queue_evidence=lambda violation_id, seconds: queued.append(violation_id),
    )(_job("job-1"))

    assert queued == []


def _reader_that_fails():
    """A reader that dies the way an expired presigned url does, mid-job."""

    def read(url, frame_range):
        raise VideoUnavailable("expired url")
        yield  # pragma: no cover — makes this a generator, as read_frames is

    return read


# --- what the source says while a job runs -----------------------------------------
#
# Detection is asynchronous, so between the 202 and the first violation the site's
# violation list is empty — and an empty list is what a site with no violations returns.
# These three writes are the only thing that tells those apart.


def test_a_job_that_finishes_leaves_its_source_completed():
    status = FakeSourceStatus()

    _handler(set_status=status)(_job("job-1"))

    assert status.transitions == ["processing", "completed"]


def test_processing_is_written_before_the_context_is_resolved():
    """The first thing, before anything slow or anything that can fail.

    A job that dies resolving its own calibration has still visibly started, which is
    what stops a run that failed early from looking like one that was never asked for.
    """
    status = FakeSourceStatus()

    def load_context(job):
        raise RuntimeError("no calibration")

    with pytest.raises(RuntimeError):
        _handler(load_context=load_context, set_status=status)(_job("job-1"))

    assert status.transitions == ["processing", "failed"]


def test_a_job_that_raises_marks_the_source_failed_and_still_raises():
    """The worker's contract is kept, not traded away.

    A raising handler stops the process — losing a job silently is worse than stopping
    loudly — and that is unchanged. The source records the failure on the way down so a
    run reads as failed rather than as still running for good.
    """
    status = FakeSourceStatus()
    handler = _handler(
        read=_reader_that_fails(), set_status=status
    )

    with pytest.raises(VideoUnavailable):
        handler(_job("job-1"))

    assert status.transitions == ["processing", "failed"]


def test_a_source_is_never_left_saying_it_is_still_being_analysed():
    # Whatever happens to a job, the run stops claiming to be in progress. This is the
    # property the front end depends on; the two tests above are how it is reached.
    for build in (
        lambda status: _handler(set_status=status),
        lambda status: _handler(read=_reader_that_fails(), set_status=status),
    ):
        status = FakeSourceStatus()
        try:
            build(status)(_job("job-1"))
        except Exception:
            pass
        assert status.transitions[-1] != "processing"
