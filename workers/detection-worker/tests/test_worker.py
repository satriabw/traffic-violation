import numpy as np
import pytest
import supervision as sv
from shared.models.detection import DetectionJob, FrameRange, JobSource, ViolationType
from shared.queue.memory import InMemoryQueue

from detection_worker.reader import VideoUnavailable
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


class FakeModel:
    """A detector with no model behind it.

    The point of the DetectionModel protocol being one method: everything downstream
    of inference can be exercised on a laptop with no GPU and no weights file.
    """

    def __init__(self, detections_per_frame: int = 0):
        self.frames: list[np.ndarray] = []
        self._count = detections_per_frame

    def predict(self, frame: np.ndarray) -> sv.Detections:
        self.frames.append(frame)
        if self._count == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array(
                [[i, i, i + 10, i + 10] for i in range(self._count)], dtype=np.float32
            ),
            confidence=np.full(self._count, 0.9, dtype=np.float32),
            class_id=np.full(self._count, 2, dtype=np.int16),
        )


class FakeTracker:
    def __init__(self, fps):
        self.fps = fps
        self.updates: list[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.updates.append(detections)
        if len(detections):
            detections.tracker_id = np.arange(1, len(detections) + 1)
        return detections


def _trackers():
    """A tracker factory that records every tracker it was asked to build."""
    made: list[FakeTracker] = []

    def factory(fps):
        tracker = FakeTracker(fps)
        made.append(tracker)
        return tracker

    factory.made = made
    return factory


def _handler(model=None, sign=None, read=None, new_tracker=None):
    return make_handler(
        model if model is not None else FakeModel(),
        sign=sign or (lambda key: "u"),
        read=read if read is not None else _reader_of(frame_count=0),
        new_tracker=new_tracker or _trackers(),
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


# --- detection and tracking ---------------------------------------------------


def test_every_frame_read_reaches_the_model():
    model = FakeModel()
    read = _reader_of(frame_count=5)

    _handler(model=model, read=read)(_job("job-0"))

    assert len(model.frames) == 5


def test_the_tracker_is_built_once_per_job_not_once_per_frame():
    # The lifetime this whole design turns on. A tracker built per frame would have no
    # memory of the previous one, so nothing would ever hold an id for two frames.
    trackers = _trackers()

    _handler(read=_reader_of(frame_count=6), new_tracker=trackers)(_job("job-0"))

    assert len(trackers.made) == 1


def test_each_job_gets_its_own_tracker():
    # The other half of it: state from one job must not survive into the next, or a
    # track from one site could be re-matched against another's.
    trackers = _trackers()
    handle = _handler(read=_reader_of(frame_count=2), new_tracker=trackers)

    handle(_job("job-0"))
    handle(_job("job-1"))

    assert len(trackers.made) == 2
    assert trackers.made[0] is not trackers.made[1]


def test_the_tracker_is_built_with_the_sources_frame_rate():
    trackers = _trackers()

    _handler(read=_reader_of(frame_count=1), new_tracker=trackers)(_job("job-0"))

    # _job carries fps=30.0 on its JobSource.
    assert trackers.made[0].fps == 30.0


def test_the_tracker_sees_what_the_model_produced():
    model = FakeModel(detections_per_frame=3)
    trackers = _trackers()

    _handler(model=model, read=_reader_of(frame_count=2), new_tracker=trackers)(_job("job-0"))

    updates = trackers.made[0].updates
    assert len(updates) == 2
    assert [len(update) for update in updates] == [3, 3]


def test_frames_with_nothing_in_them_still_reach_the_tracker():
    # Most frames of most footage. Skipping the update would let the tracker's frame
    # counter fall behind the video and age every lost track wrongly.
    model = FakeModel(detections_per_frame=0)
    trackers = _trackers()

    _handler(model=model, read=_reader_of(frame_count=4), new_tracker=trackers)(_job("job-0"))

    assert len(trackers.made[0].updates) == 4


def test_the_summary_logs_how_much_was_detected_and_tracked(caplog):
    model = FakeModel(detections_per_frame=2)

    with caplog.at_level("INFO"):
        _handler(model=model, read=_reader_of(frame_count=3))(_job("job-0"))

    # 3 frames x 2 detections, and the fake tracker numbers them 1..2 every frame, so
    # two distinct ids across the job.
    assert "read=3 detections=6 tracks=2" in caplog.text


def test_a_job_that_detects_nothing_still_logs_a_summary(caplog):
    with caplog.at_level("INFO"):
        _handler(model=FakeModel(0), read=_reader_of(frame_count=3))(_job("job-0"))

    assert "read=3 detections=0 tracks=0" in caplog.text


def test_per_frame_detail_is_logged_at_debug_not_info(caplog):
    # A 30-second chunk is ~900 frames. Per-frame lines at INFO would bury the one
    # line anyone watching a normal run actually wants.
    model = FakeModel(detections_per_frame=1)

    with caplog.at_level("INFO"):
        _handler(model=model, read=_reader_of(frame_count=2))(_job("job-0"))
    assert "frame 0" not in caplog.text

    caplog.clear()
    with caplog.at_level("DEBUG"):
        _handler(model=model, read=_reader_of(frame_count=2))(_job("job-0"))
    assert "frame 0 detections=1 ids=[1]" in caplog.text
