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
from evidence_collector import EvidenceCollector, ObjectState, TrackWindow
from trajectory_collector import NullCollector, Trajectory, TrajectoryCollector
from violation_detector import (
    Configuration,
    Detector,
    TrackedObject,
    Violation,
    get_detector,
)

# How much lead-up an analyzer built by hand keeps. A caller that resolved a site's
# context has a number from it and passes that instead; this is only what a bare
# constructor falls back to. Imported rather than restated, so there is one default,
# in the module that reads the document it comes from.
from detection_worker.context import DEFAULT_EVIDENCE_SECONDS
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


def _object_states(
    tracked_objects: Iterable[TrackedObject],
    trajectories: dict[int, Trajectory],
) -> list[ObjectState]:
    """A frame's tracked objects and their trajectories, joined for the record.

    The third adapter, and the last of the boundary. Where `_tracked` and
    `_tracked_objects` take one library's type apart for one package, this one puts two
    packages' answers back together — the box a rule reasoned about and the ground
    position a collector measured, which are separate exactly because neither package
    knows about the other.

    A track with no trajectory keeps None for both position and speed rather than being
    dropped. Its box is still evidence: an object above the horizon, or one in its
    filter's first frames, was visibly there and the record should say so. Deciding
    what to do with a missing position is the evidence package's job, and it decides it
    per window rather than per frame.
    """
    return [
        ObjectState(
            track_id=tracked.track_id,
            bbox=tracked.bbox,
            class_name=tracked.class_name,
            position=trajectory.position if trajectory else None,
            speed=trajectory.speed if trajectory else None,
        )
        for tracked in tracked_objects
        for trajectory in [trajectories.get(tracked.track_id)]
    ]


class FrameAnalyzer:
    """Detect, track, locate. The rule engine belongs after the last of those."""

    def __init__(
        self,
        model: DetectionModel,
        tracker: Tracker,
        trajectory_collector: TrajectoryCollector = NullCollector(),
        detector: Detector = Detector(),
        evidence: EvidenceCollector | None = None,
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
        # None, and built here — the one collaborator that does not follow the
        # "always, never None" rule above it. That rule works for the other two because
        # their defaults hold no state and can therefore be shared by every analyzer
        # that takes one; an evidence collector is a ring of the last few seconds, so a
        # shared default would splice one job's footage into the next one's records.
        # Defaulted at all rather than required so a test building an analyzer by hand
        # does not have to know how long the window is.
        #
        # `is None`, NOT `or`. An evidence collector defines `__len__`, so one that has
        # not recorded anything yet — which is every one of them at this point — is
        # falsy, and `or` would throw away the collector it was handed and build a
        # default in its place. Silently: the job would run, record everything, and
        # report windows of the wrong length in the wrong coordinates.
        self._evidence = (
            EvidenceCollector.over(seconds=DEFAULT_EVIDENCE_SECONDS, fps=resolve_fps(None))
            if evidence is None
            else evidence
        )

    def analyze(self, frame: np.ndarray, index: int) -> FrameResult:
        detections = self._model.predict(frame)
        # Every frame, empty or not: the tracker counts frames by counting updates, and
        # skipping one ages every lost track wrongly.
        tracked = self._tracker.update(detections)

        # Built once and used twice: the rules reason about these, and the record keeps
        # them. Building them a second time for the record would be the same work done
        # again to produce the same answer.
        tracked_objects = _tracked_objects(tracked)

        boxes, track_ids = _tracked(tracked)
        # The absolute index, not a running count of calls. The collector measures how
        # long a track was missing for, and a job covering frames 900-1800 would
        # otherwise have every gap measured against a counter starting at zero.
        trajectories = self._trajectory_collector.collect(boxes, track_ids, index)

        # EVERY FRAME, and before the rules run. A window is a duration, so a frame
        # with nothing on it is as much a part of the record as one with a car in it —
        # four seconds of an empty crossing is why the fifth matters. Before, because
        # the frame a rule fires on has to be in the window it reads: the violation is
        # the end of the story, not something that happened after it.
        self._evidence.observe(index, _object_states(tracked_objects, trajectories))

        # After tracking, never before: every rule in the package is about what an
        # object did over time, and an untracked box has no history to have done it in.
        violations = self._detector.detect(frame, tracked_objects, index)

        return FrameResult(
            index=index,
            detections=tracked,
            trajectories=trajectories,
            violations=violations,
            evidence=self._evidence_for(violations),
        )

    def _evidence_for(self, violations: list[Violation]) -> dict[int, TrackWindow]:
        """The lead-up to whatever just fired, keyed by the track it fired on.

        Nothing at all on the overwhelming majority of frames, where nothing fires —
        and reading a window costs nothing on those, because there is nothing to ask
        for.

        Keyed by track id rather than aligned with `violations`, so two rules
        convicting one vehicle on one frame share the one history they both describe
        rather than carrying a copy each.
        """
        if not violations:
            return {}
        return {
            window.track_id: window
            for window in self._evidence.window_for({v.track_id for v in violations})
        }

    @property
    def evidence_capacity(self) -> int:
        """How many frames of lead-up this job's record can hold at most.

        Exposed so the handler can tell a window that is short because the track was
        only just seen from one that is short because the chunk started too late. The
        difference does not show in the window itself.
        """
        return self._evidence.capacity

    def finish(self) -> list[Violation]:
        """Whatever the rules were still holding when the frames ran out.

        Call this once, after the last `analyze`. Empty for every rule that ships
        today, because a rule decides on the frame it is given — but a module working
        on a clip is always holding a partial one, and without this the last seconds of
        every job would be dropped in silence.

        The violations it returns carry their own `frame_index`, which for a buffering
        module is earlier than the last frame analysed. Recording the loop's index
        instead would misdate them by the length of the window.

        NO EVIDENCE COMES BACK WITH THESE, and it is the same buffering that is the
        reason. A module flushing a partial clip reports on a frame the ring has
        already rolled past, so the window it would be handed describes the end of the
        job rather than the moment convicted — worse than none, because it would look
        right. Empty today, since every rule that ships decides on the frame it is
        given. The fix when a clip module lands is for it to say how far back it
        reasons, and to size the ring to cover it.
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
    seconds: float = DEFAULT_EVIDENCE_SECONDS,
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
    # A third thing that scales by the frame rate, and the reason `resolve_fps` is one
    # definition: a job whose tracker aged lost tracks at 30 while its evidence ring
    # held five seconds of something else would disagree with itself about how much
    # time a frame is worth.
    #
    # It is told nothing about the calibration, and needs to be told nothing. A job
    # without one produces no trajectories, so the record simply has no positions in
    # it — the same absence, arrived at without anyone having to decide anything.
    #
    # `seconds` arrives as a number. Which number is the site's decision, taken where
    # its configuration document was resolved — nothing here is handed the document to
    # go looking in.
    evidence = EvidenceCollector.over(seconds=seconds, fps=resolved)
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
    return FrameAnalyzer(model, new_tracker(resolved), collector, detector, evidence)
