import numpy as np
import pytest
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

    make_handler(sign=lambda key: signed.append(key) or f"https://r2/{key}?sig", read=read)(
        _job("job-0", key="video/file-9/clip.mp4")
    )

    assert signed == ["video/file-9/clip.mp4"]


def test_the_handler_reads_the_signed_url_over_the_job_s_frame_range():
    read = _reader_of(frame_count=0)

    make_handler(sign=lambda key: "https://r2/signed", read=read)(
        _job("job-0", start=900, end=1800)
    )

    url, frame_range = read.calls[0]
    assert url == "https://r2/signed"
    assert (frame_range.start, frame_range.end) == (900, 1800)


def test_the_handler_reads_every_frame_and_logs_the_count(caplog):
    read = _reader_of(frame_count=4)

    with caplog.at_level("INFO"):
        make_handler(sign=lambda key: "u", read=read)(_job("job-0"))

    # Counting the frames is the whole job for now; the count is the evidence the
    # decode actually happened, so it has to reach the log.
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
        run(queue, handle=make_handler(sign=lambda key: "u", read=read))

    # The second job is still queued: nothing was silently consumed on the way out.
    assert queue.consume().id == "job-1"
