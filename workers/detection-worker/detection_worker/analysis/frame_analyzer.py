"""The per-frame pipeline: one frame in, everything that frame produced out.

This is the half of the old handler that runs once per frame. What is left in
`worker.py` is the half that runs once per job — sign a url, iterate the reader,
aggregate, log — and the split is the point. The rule engine lands inside `analyze`,
and it has no business widening a function that also knows about presigned urls.

An analyzer belongs to exactly ONE job. It holds a tracker, which holds live state
(active tracks, lost tracks, a Kalman filter per track, a frame counter), so sharing one
across jobs would let a track from one site's chunk be re-matched against another's.
That is the same lifetime `make_tracker` already enforced; the analyzer inherits it
rather than inventing a second rule. The trajectory collector wants it for the same
reason and by the same key — tracker ids restart at 1 for every job, so a collector
outliving one would merge unrelated objects.
"""

from typing import Any, Callable, Mapping

import numpy as np
import supervision as sv
from trajectory_collector import NullCollector, TrajectoryCollector

from detection_worker.analysis.frame_result import FrameResult
from detection_worker.detection.model import DetectionModel
from detection_worker.detection.tracker import Tracker, make_tracker, resolve_fps


def _tracked(detections: sv.Detections) -> tuple[np.ndarray, np.ndarray]:
    """`sv.Detections` down to the two arrays a collector takes.

    This is the whole of the adapter between the detection library and the trajectory
    package, and it is deliberately this small. The package takes arrays rather than
    anything of supervision's so that it depends on nothing this worker depends on —
    which is what makes it liftable out of this repository — and the cost of that is
    exactly these two attribute reads.

    `tracker_id` is None rather than an empty array on a frame the tracker had nothing
    to assign, so an empty frame becomes empty arrays rather than a crash.
    """
    if detections.tracker_id is None:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=int)
    return detections.xyxy, detections.tracker_id


class FrameAnalyzer:
    """Detect, track, locate. The rule engine belongs after the last of those."""

    def __init__(
        self,
        model: DetectionModel,
        tracker: Tracker,
        trajectory_collector: TrajectoryCollector = NullCollector(),
    ):
        self._model = model
        self._tracker = tracker
        # A collector, always — never None. A site with no calibration gets one that
        # reports nothing, so `analyze` has no branch and `FrameResult.trajectories` is
        # a dict on every frame rather than sometimes-a-dict. The uncalibrated case is
        # normal enough to be worth designing the null check out of. Sharing one
        # NullCollector across every analyzer that defaults to it is safe for the same
        # reason it reports nothing: it holds no state to share.
        self._trajectory_collector = trajectory_collector

    def analyze(self, frame: np.ndarray, index: int) -> FrameResult:
        detections = self._model.predict(frame)
        # Every frame, empty or not: the tracker counts frames by counting updates, and
        # skipping one ages every lost track wrongly.
        tracked = self._tracker.update(detections)

        boxes, track_ids = _tracked(tracked)
        # The absolute index, not a running count of calls. The collector measures how
        # long a track was missing for, and a job covering frames 900-1800 would
        # otherwise have every gap measured against a counter starting at zero.
        trajectories = self._trajectory_collector.collect(boxes, track_ids, index)

        return FrameResult(index=index, detections=tracked, trajectories=trajectories)


def make_analyzer(
    model: DetectionModel,
    fps: float | None,
    calibration: Mapping[str, Any] | None = None,
    new_tracker: Callable[[float | None], Tracker] = make_tracker,
    new_collector: Callable[..., TrajectoryCollector] = TrajectoryCollector.from_calibration,
) -> FrameAnalyzer:
    """An analyzer for one job.

    `model` comes from outside because building it is expensive and it holds no per-job
    state — one session serves the whole process. Everything else here is cheap and
    per-job, which is why this is a factory at all.

    The frame rate is resolved once and given to both collaborators, so a job whose
    source had no probed fps cannot end up with a tracker aging tracks at one rate and
    a collector measuring gaps at another.

    `calibration` is the document the job was pinned to, already fetched — resolving it
    is `detection_worker.context`'s job, and by the time anything gets here the version
    it belongs to has been settled. None means the site had none, which is a normal
    state: detection and tracking still run, and the job simply reports no trajectories.

    A calibration that cannot be projected with raises out of here, before a single
    frame is decoded. That stops the worker, which is the same thing any other failing
    job does today — and much better than a run that produced plausible-looking
    positions from a broken camera model.
    """
    resolved = resolve_fps(fps)
    collector = (
        NullCollector()
        if calibration is None
        else new_collector(calibration, fps=resolved)
    )
    return FrameAnalyzer(model, new_tracker(resolved), collector)
