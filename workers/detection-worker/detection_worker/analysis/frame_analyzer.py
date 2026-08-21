"""The per-frame pipeline: one frame in, everything that frame produced out.

This is the half of the old handler that runs once per frame. What is left in
`worker.py` is the half that runs once per job — sign a url, iterate the reader,
aggregate, log — and the split is the point. Trajectory collection and the rule engine
both land inside `analyze`, and neither of them has any business widening a function
that also knows about presigned urls.

An analyzer belongs to exactly ONE job. It holds a tracker, which holds live state
(active tracks, lost tracks, a Kalman filter per track, a frame counter), so sharing one
across jobs would let a track from one site's chunk be re-matched against another's.
That is the same lifetime `make_tracker` already enforced; the analyzer inherits it
rather than inventing a second rule, and the collaborators arriving later — a
trajectory collector keyed by tracker id, violation modules with per-track caches —
all want exactly that lifetime too.
"""

from typing import Callable

import numpy as np

from detection_worker.analysis.frame_result import FrameResult
from detection_worker.detection.model import DetectionModel
from detection_worker.detection.tracker import Tracker, make_tracker


class FrameAnalyzer:
    """Detect, then track. More steps are coming; they belong between these two."""

    def __init__(self, model: DetectionModel, tracker: Tracker):
        self._model = model
        self._tracker = tracker

    def analyze(self, frame: np.ndarray, index: int) -> FrameResult:
        detections = self._model.predict(frame)
        # Every frame, empty or not: the tracker counts frames by counting updates, and
        # skipping one ages every lost track wrongly.
        tracked = self._tracker.update(detections)
        return FrameResult(index=index, detections=tracked)


def make_analyzer(
    model: DetectionModel,
    fps: float | None,
    new_tracker: Callable[[float | None], Tracker] = make_tracker,
) -> FrameAnalyzer:
    """An analyzer for one job.

    `model` comes from outside because building it is expensive and it holds no per-job
    state — one session serves the whole process. Everything else here is cheap and
    per-job, which is why this is a factory at all.

    `fps` is passed straight through: what an unknown frame rate falls back to is the
    tracker's decision, not this one's.
    """
    return FrameAnalyzer(model, new_tracker(fps))
