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

from typing import Callable, Iterable

import numpy as np
import supervision as sv
from trajectory_collector import NullCollector, TrajectoryCollector
from violation_detector import (
    Configuration,
    Detector,
    TrackedObject,
    Violation,
    get_detector,
)

from detection_worker.analysis.frame_result import FrameResult
from detection_worker.detection.model import DetectionModel
from detection_worker.detection.tracker import Tracker, make_tracker, resolve_fps

# Where the detector puts the name it gave each box. See detection.model — an id
# outside its class map still yields a detection, named by its number.
CLASS_NAME = "class_name"


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


def _class_names(detections: sv.Detections) -> list[str]:
    """What the detector called each box.

    NAMES, NOT IDS, and this is the one place that has to know the difference. The
    model puts a name beside every box precisely so nothing downstream needs its class
    map, and a rule package that took ids would need to be told which model produced
    them. Falling back to the id as a string mirrors what the model already does for an
    id outside its own map: a rule has no opinion about "9", which is the correct
    amount of opinion to have about a class nobody named.
    """
    names = detections.data.get(CLASS_NAME)
    if names is not None:
        return [str(name) for name in names]
    if detections.class_id is None:
        return [""] * len(detections)
    return [str(int(class_id)) for class_id in detections.class_id]


def _tracked_objects(detections: sv.Detections) -> list[TrackedObject]:
    """`sv.Detections` down to the list a violation detector takes.

    The counterpart of `_tracked` above, and small for the same reason: the rule
    package takes plain numbers and strings rather than anything of supervision's, so
    that it depends on nothing this worker depends on. Two adapters rather than one
    because the two packages want genuinely different things — a collector projects
    boxes and does not care what they are, a rule cares about little else.
    """
    if detections.tracker_id is None:
        return []
    return [
        TrackedObject(
            track_id=int(track_id),
            bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            class_name=class_name,
        )
        for track_id, box, class_name in zip(
            detections.tracker_id, detections.xyxy, _class_names(detections)
        )
    ]


class FrameAnalyzer:
    """Detect, track, locate. The rule engine belongs after the last of those."""

    def __init__(
        self,
        model: DetectionModel,
        tracker: Tracker,
        trajectory_collector: TrajectoryCollector = NullCollector(),
        detector: Detector = Detector(),
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
        # A detector, always — never None, on exactly the terms above. A site with no
        # configuration gets one carrying no rules, so `analyze` has no branch in it
        # and `FrameResult.violations` is a list on every frame. Sharing one empty
        # Detector across every analyzer that defaults to it is safe because it holds
        # no rules and therefore no state; a detector built from a document is not
        # shareable, and `make_analyzer` builds one per job.
        self._detector = detector

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

        # After tracking, never before: every rule in the package is about what an
        # object did over time, and an untracked box has no history to have done it in.
        violations = self._detector.detect(frame, _tracked_objects(tracked), index)

        return FrameResult(
            index=index,
            detections=tracked,
            trajectories=trajectories,
            violations=violations,
        )

    def finish(self) -> list[Violation]:
        """Whatever the rules were still holding when the frames ran out.

        Call this once, after the last `analyze`. Empty for every rule that ships
        today, because a rule decides on the frame it is given — but a module working
        on a clip is always holding a partial one, and without this the last seconds of
        every job would be dropped in silence.

        The violations it returns carry their own `frame_index`, which for a buffering
        module is earlier than the last frame analysed. Recording the loop's index
        instead would misdate them by the length of the window.
        """
        return self._detector.finish()


def make_analyzer(
    model: DetectionModel,
    fps: float | None,
    calibration: bytes | None = None,
    configuration: dict | None = None,
    types: Iterable[str] | None = None,
    new_tracker: Callable[[float | None], Tracker] = make_tracker,
    new_collector: Callable[..., TrajectoryCollector] = TrajectoryCollector.from_calibration,
    new_detector: Callable[..., Detector] = get_detector,
) -> FrameAnalyzer:
    """An analyzer for one job.

    `model` comes from outside because building it is expensive and it holds no per-job
    state — one session serves the whole process. Everything else here is cheap and
    per-job, which is why this is a factory at all.

    The frame rate is resolved once and given to both collaborators, so a job whose
    source had no probed fps cannot end up with a tracker aging tracks at one rate and
    a collector measuring gaps at another.

    `calibration` is the raw document the job was pinned to, as fetched — resolving
    *which* document is `detection_worker.context`'s job, and by the time anything gets
    here the version it belongs to has been settled. What is inside it is the trajectory
    package's business: a calibration is an OpenCV FileStorage document, and nothing on
    this side of the boundary should have an opinion about that. None means
    the site had none, which is a normal state: detection and tracking still run, and
    the job simply reports no trajectories.

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
    detector = (
        Detector()
        if configuration is None
        else new_detector(
            Configuration.from_document(configuration), types=types, fps=resolved
        )
    )
    return FrameAnalyzer(model, new_tracker(resolved), collector, detector)
