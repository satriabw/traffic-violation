"""Ground-plane trajectories from tracked bounding boxes.

    from trajectory_collector import TrajectoryCollector

    collector = TrajectoryCollector.from_calibration("camera_model.yml", fps=30.0)
    trajectories = collector.collect(boxes, track_ids, frame_index)

Everything below this line is an implementation detail — the camera model, the filter,
which projection a calibration calls for. Callers name one class.

The re-exports here are deliberate, and a departure from how packages inside this
repository are laid out: an application's `__init__.py` stays empty because its modules
are imported by path, but a library's import path is its API. Moving a module should
not break anyone.
"""

from trajectory_collector.camera_model import CalibrationInvalid, CameraModel
from trajectory_collector.collector import (
    NullCollector,
    Trajectory,
    TrajectoryCollector,
)
from trajectory_collector.kalman import FilterParams
from trajectory_collector.pinhole import CollectorParams, PinholeCollector

__all__ = [
    "CalibrationInvalid",
    "CameraModel",
    "CollectorParams",
    "FilterParams",
    "NullCollector",
    "PinholeCollector",
    "Trajectory",
    "TrajectoryCollector",
]
