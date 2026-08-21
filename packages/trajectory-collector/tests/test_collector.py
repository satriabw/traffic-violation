import dataclasses

import numpy as np
import pytest

from trajectory_collector import (
    CalibrationInvalid,
    NullCollector,
    Trajectory,
    TrajectoryCollector,
)


def test_a_trajectory_is_a_position_and_a_speed():
    trajectory = Trajectory(position=(12.5, -3.0), speed=8.25)

    assert trajectory.position == (12.5, -3.0)
    assert trajectory.speed == 8.25


def test_a_trajectory_cannot_be_edited_after_the_fact():
    # It is a reading, not a running total. A caller that could rewrite one could
    # change what a violation was recorded against after the fact.
    trajectory = Trajectory(position=(0.0, 0.0), speed=0.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.speed = 30.0


def test_a_collector_must_implement_collect():
    # The one method the interface is for. A subclass that forgets it should fail at
    # construction rather than on some frame in the middle of a video.
    class Incomplete(TrajectoryCollector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_from_calibration_builds_the_collector_a_calibration_calls_for():
    # The one entry point. Which collector comes back is the package's decision, taken
    # from what the document contains — callers name this class and no other.
    collector = TrajectoryCollector.from_calibration(
        {
            "camera_matrix": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
            "rot_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "tvec": [0.0, 0.0, 100.0],
        },
        fps=30.0,
    )

    assert isinstance(collector, TrajectoryCollector)
    assert not isinstance(collector, NullCollector)


def test_from_calibration_rejects_a_document_it_cannot_project_with():
    # At construction, not on some frame in the middle of a video.
    with pytest.raises(CalibrationInvalid):
        TrajectoryCollector.from_calibration({"camera_matrix": []}, fps=30.0)


# --- the no-calibration case --------------------------------------------------


def _boxes(count: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.array(
        [[i, i, i + 10, i + 20] for i in range(count)], dtype=np.float32
    )
    return boxes, np.arange(1, count + 1)


def test_a_null_collector_reports_nothing_for_a_frame_with_objects_in_it():
    # Not an error, and not a guess either: with no calibration there is no ground
    # plane, so there is no honest position to report for anything.
    boxes, track_ids = _boxes(3)

    assert NullCollector().collect(boxes, track_ids, frame_index=0) == {}


def test_a_null_collector_reports_nothing_for_an_empty_frame():
    collector = NullCollector()

    empty = np.empty((0, 4), dtype=np.float32)
    assert collector.collect(empty, np.empty(0, dtype=int), frame_index=0) == {}


def test_a_null_collector_is_a_collector():
    # It stands in wherever a real one goes, which is the point — the caller keeps one
    # code path instead of a null check per frame.
    assert isinstance(NullCollector(), TrajectoryCollector)
