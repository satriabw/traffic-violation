from shared.models.detection import DetectionJob, FrameRange, ViolationType
from shared.queue.memory import InMemoryQueue

from detection_worker.worker import run


def _job(job_id: str) -> DetectionJob:
    return DetectionJob(
        id=job_id,
        site_id="site-1",
        frame_range=FrameRange(start=0, end=100),
        types=[ViolationType.RED_LIGHT_RUNNING],
    )


def _queue_of(*job_ids: str) -> InMemoryQueue:
    queue = InMemoryQueue()
    for job_id in job_ids:
        queue.enqueue(_job(job_id))
    return queue


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


def test_the_default_handler_consumes_a_job_without_raising():
    # The stub's whole job: arrive, be logged, not crash.
    assert run(_queue_of("job-0")) == 1
