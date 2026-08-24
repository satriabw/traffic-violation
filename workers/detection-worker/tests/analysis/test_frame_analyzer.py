import numpy as np
import pytest
import supervision as sv
from trajectory_collector import (
    CalibrationInvalid,
    NullCollector,
    Trajectory,
    TrajectoryCollector,
)

from violation_detector import (
    ConfigurationInvalid,
    Detector,
    TrackedObject,
    Violation,
)

from detection_worker.analysis.frame_analyzer import (
    FrameAnalyzer,
    _tracked_objects,
    make_analyzer,
)
from detection_worker.detection.tracker import DEFAULT_FPS


class FakeModel:
    """A detector with no model behind it.

    The point of the DetectionModel protocol being one method: everything downstream
    of inference can be exercised on a laptop with no GPU and no weights file.
    """

    def __init__(self, detections_per_frame: int = 0, names: list[str] | None = None):
        self.frames: list[np.ndarray] = []
        self._count = detections_per_frame
        # None means no `class_name` in `data` at all, which is what a detection built
        # by hand looks like. The real model always supplies one.
        self._names = names

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
            data={} if self._names is None else {"class_name": np.array(self._names)},
        )


class FakeTracker:
    """Records every update, and numbers whatever it is given 1..n."""

    def __init__(self, fps=None):
        self.fps = fps
        self.updates: list[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.updates.append(detections)
        if len(detections):
            detections.tracker_id = np.arange(1, len(detections) + 1)
        return detections


class FakeCollector(TrajectoryCollector):
    """Records what it was asked to locate, and names one trajectory per track."""

    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def collect(self, boxes, track_ids, frame_index):
        self.calls.append((boxes, track_ids, frame_index))
        return {
            int(track_id): Trajectory(position=(float(track_id), 0.0), speed=1.0)
            for track_id in track_ids
        }


def _trackers():
    """A tracker factory that records every tracker it was asked to build."""
    made: list[FakeTracker] = []

    def factory(fps):
        tracker = FakeTracker(fps)
        made.append(tracker)
        return tracker

    factory.made = made
    return factory


def _frame() -> np.ndarray:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_the_frame_reaches_the_model():
    model = FakeModel()
    FrameAnalyzer(model, FakeTracker()).analyze(_frame(), index=0)

    assert len(model.frames) == 1


def test_what_the_model_produced_reaches_the_tracker():
    tracker = FakeTracker()

    FrameAnalyzer(FakeModel(detections_per_frame=3), tracker).analyze(_frame(), index=0)

    assert [len(update) for update in tracker.updates] == [3]


def test_a_frame_with_nothing_in_it_still_reaches_the_tracker():
    # Most frames of most footage. Skipping the update would let the tracker's frame
    # counter fall behind the video and age every lost track wrongly.
    tracker = FakeTracker()

    FrameAnalyzer(FakeModel(detections_per_frame=0), tracker).analyze(_frame(), index=0)

    assert len(tracker.updates) == 1


def test_the_result_carries_the_tracked_detections_not_the_raw_ones():
    result = FrameAnalyzer(FakeModel(detections_per_frame=2), FakeTracker()).analyze(
        _frame(), index=0
    )

    # Raw detections have no tracker_id at all; these came back through the tracker.
    assert result.detections.tracker_id is not None
    assert result.track_ids == [1, 2]


def test_the_result_carries_the_index_it_was_given():
    # Absolute, not the position in this run: a job covering 900-1800 reports 900 for
    # its first frame, because that is the number a violation is recorded against.
    result = FrameAnalyzer(FakeModel(), FakeTracker()).analyze(_frame(), index=900)

    assert result.index == 900


def test_a_frame_with_no_detections_has_no_track_ids():
    # tracker_id is None rather than an empty array, which is what track_ids exists
    # to hide from everyone downstream.
    result = FrameAnalyzer(FakeModel(0), FakeTracker()).analyze(_frame(), index=0)

    assert result.track_ids == []


def test_one_analyzer_keeps_one_tracker_across_frames():
    # The lifetime this whole design turns on. A tracker rebuilt per frame would have
    # no memory of the previous one, so nothing would ever hold an id for two frames.
    tracker = FakeTracker()
    analyzer = FrameAnalyzer(FakeModel(detections_per_frame=1), tracker)

    for index in range(4):
        analyzer.analyze(_frame(), index)

    assert len(tracker.updates) == 4


def test_make_analyzer_builds_a_tracker_at_the_sources_frame_rate():
    # Passed straight through — what an unknown or zero rate falls back to is
    # make_tracker's decision, and its own suite covers it.
    trackers = _trackers()

    make_analyzer(FakeModel(), fps=25.0, new_tracker=trackers)

    assert trackers.made[0].fps == 25.0


def test_each_call_to_make_analyzer_builds_its_own_tracker():
    # State from one job must not survive into the next, or a track from one site
    # could be re-matched against another's.
    trackers = _trackers()
    model = FakeModel()

    make_analyzer(model, 30.0, new_tracker=trackers)
    make_analyzer(model, 30.0, new_tracker=trackers)

    assert len(trackers.made) == 2
    assert trackers.made[0] is not trackers.made[1]


# --- trajectories -------------------------------------------------------------


def test_the_tracked_boxes_and_ids_reach_the_collector():
    # Arrays, not sv.Detections: the trajectory package depends on nothing this worker
    # depends on, and this two-attribute translation is the entire price of that.
    collector = FakeCollector()

    FrameAnalyzer(FakeModel(detections_per_frame=2), FakeTracker(), collector).analyze(
        _frame(), index=0
    )

    boxes, track_ids, _ = collector.calls[0]
    assert boxes.shape == (2, 4)
    assert list(track_ids) == [1, 2]


def test_the_collector_is_given_the_absolute_frame_index():
    # Not a count of calls. The collector measures how long a track was missing for,
    # and a job covering 900-1800 would otherwise measure every gap against zero.
    collector = FakeCollector()
    analyzer = FrameAnalyzer(FakeModel(detections_per_frame=1), FakeTracker(), collector)

    analyzer.analyze(_frame(), index=900)
    analyzer.analyze(_frame(), index=901)

    assert [frame_index for _, _, frame_index in collector.calls] == [900, 901]


def test_what_the_collector_returned_reaches_the_result():
    result = FrameAnalyzer(
        FakeModel(detections_per_frame=2), FakeTracker(), FakeCollector()
    ).analyze(_frame(), index=0)

    assert result.trajectories == {
        1: Trajectory(position=(1.0, 0.0), speed=1.0),
        2: Trajectory(position=(2.0, 0.0), speed=1.0),
    }


def test_a_frame_with_nothing_tracked_still_reaches_the_collector():
    # tracker_id is None on such a frame, which becomes empty arrays rather than a
    # crash — and the collector still hears about the frame, because a frame in which
    # a track was absent is exactly what it needs to measure a gap.
    collector = FakeCollector()

    FrameAnalyzer(FakeModel(detections_per_frame=0), FakeTracker(), collector).analyze(
        _frame(), index=7
    )

    boxes, track_ids, frame_index = collector.calls[0]
    assert boxes.shape == (0, 4)
    assert len(track_ids) == 0
    assert frame_index == 7


def test_an_analyzer_with_no_collector_reports_no_trajectories():
    # The uncalibrated site: normal, not an error. There is no ground plane to project
    # onto, so there is no honest position to report.
    result = FrameAnalyzer(FakeModel(detections_per_frame=2), FakeTracker()).analyze(
        _frame(), index=0
    )

    assert result.trajectories == {}


def test_a_job_with_no_calibration_gets_a_collector_that_reports_nothing():
    # A site with no calibration is a normal state, not a failure. Detection and
    # tracking still run; there is simply no ground plane to put anything on.
    analyzer = make_analyzer(FakeModel(), fps=30.0, new_tracker=_trackers())

    assert isinstance(analyzer._trajectory_collector, NullCollector)


def test_a_job_with_a_calibration_gets_a_collector_built_from_it():
    built: list[tuple] = []
    collector = FakeCollector()

    analyzer = make_analyzer(
        FakeModel(),
        fps=30.0,
        calibration={"camera_matrix": "v3"},
        new_tracker=_trackers(),
        new_collector=lambda document, fps: built.append((document, fps)) or collector,
    )

    assert built == [({"camera_matrix": "v3"}, 30.0)]
    assert analyzer._trajectory_collector is collector


def test_the_tracker_and_the_collector_are_given_the_same_frame_rate():
    # A source ffprobe could not read. Both fall back, and they have to fall back to
    # the same number — one aging lost tracks at 30fps while the other measures gaps at
    # some other rate would have the two disagree about how much time a frame is worth.
    trackers = _trackers()
    built: list[float] = []

    make_analyzer(
        FakeModel(),
        fps=None,
        calibration={"camera_matrix": "v3"},
        new_tracker=trackers,
        new_collector=lambda document, fps: built.append(fps) or FakeCollector(),
    )

    assert trackers.made[0].fps == built[0] == DEFAULT_FPS


def test_a_calibration_that_cannot_be_projected_with_stops_the_job():
    # Before a frame is decoded, and loudly. A run that produced plausible-looking
    # positions from a broken camera model is far worse than one that refused to start.
    def new_collector(document, fps):
        raise CalibrationInvalid("camera_matrix has shape (0,), expected (3, 3)")

    with pytest.raises(CalibrationInvalid):
        make_analyzer(
            FakeModel(),
            fps=30.0,
            calibration={"camera_matrix": []},
            new_tracker=_trackers(),
            new_collector=new_collector,
        )


# --- the calibration format ---------------------------------------------------

OPENCV_YML = b"""%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 0., 0., 1000., 0., 0., 0., 1. ]
rot_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
tvec: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0., 0., 100. ]
"""


def test_a_job_whose_calibration_is_an_opencv_document_locates_its_tracks():
    # The format calibrations are written in. It reaches the analyzer as the raw bytes
    # context fetched, and nothing on this side of the boundary parses it — that is the
    # trajectory package's business.
    analyzer = make_analyzer(
        FakeModel(detections_per_frame=1), fps=30.0, calibration=OPENCV_YML
    )

    result = analyzer.analyze(np.zeros((720, 1280, 3), dtype=np.uint8), index=0)

    # The fake model puts one box at (0, 0, 10, 10), so its anchor is (5, 10) — a
    # hundredth of the 1000px focal length, seen from 100m up, is 0.5m by 1m.
    assert result.trajectories[1].position == pytest.approx((0.5, 1.0))


def test_a_calibration_document_that_is_neither_format_stops_the_job():
    with pytest.raises(CalibrationInvalid):
        make_analyzer(FakeModel(), fps=30.0, calibration=b"not a calibration")


# --- violations ---------------------------------------------------------------

# A junction, small but complete. Only `make_analyzer` reads it; the tests that drive
# the analyzer directly hand it a fake detector instead.
CONFIGURATION = {
    "version": 1,
    "violations": ["rlr_violation"],
    "regions": {
        "lanes": [{"id": "lane_1", "points": [[100, 200], [200, 200], [200, 400], [100, 400]]}],
        "rois": [{"id": "roi_1", "points": [[100, 100], [200, 100], [200, 200], [100, 200]]}],
        "traffic_lights": [
            {"id": "tl_1", "points": [[10, 10], [25, 10], [25, 55]], "controls": ["lane_1"]}
        ],
    },
}


class FakeDetector(Detector):
    """Records what it was asked about, and reports one violation per object."""

    def __init__(self, held: list[Violation] | None = None):
        super().__init__()
        self.calls: list[tuple[list[TrackedObject], int]] = []
        self._held = held or []

    def detect(self, frame, tracked_objects, frame_index):
        self.calls.append((tracked_objects, frame_index))
        return [
            Violation(type="red_light_running", track_id=object.track_id, frame_index=frame_index)
            for object in tracked_objects
        ]

    def finish(self):
        return self._held


def test_the_adapter_carries_the_name_the_model_gave_each_box():
    # Names, not ids. This is the one place that has to know the difference, and the
    # reason the rule package never needs a class map.
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int16),
        tracker_id=np.array([7]),
        data={"class_name": np.array(["motorbike"])},
    )

    assert _tracked_objects(detections) == [
        TrackedObject(track_id=7, bbox=(0.0, 0.0, 10.0, 20.0), class_name="motorbike")
    ]


def test_the_adapter_falls_back_to_the_id_when_nothing_named_the_box():
    # What the model already does for an id outside its own map. A rule has no opinion
    # about "3", which is the correct amount of opinion to have.
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int16),
        tracker_id=np.array([7]),
    )

    assert _tracked_objects(detections)[0].class_name == "3"


def test_the_adapter_reports_nothing_for_an_untracked_frame():
    # tracker_id is None rather than an empty array on such a frame, which is most
    # frames of most footage.
    assert _tracked_objects(sv.Detections.empty()) == []


def test_the_tracked_objects_reach_the_detector():
    # After tracking, never before: every rule is about what an object did over time,
    # and an untracked box has no history to have done it in.
    detector = FakeDetector()

    FrameAnalyzer(
        FakeModel(detections_per_frame=2, names=["car", "person"]),
        FakeTracker(),
        NullCollector(),
        detector,
    ).analyze(_frame(), index=0)

    tracked_objects, _ = detector.calls[0]
    assert [(o.track_id, o.class_name) for o in tracked_objects] == [(1, "car"), (2, "person")]


def test_the_detector_is_given_the_absolute_frame_index():
    # The number a violation is recorded against, so a reviewer can find it in the
    # footage. A running count of calls would point at the wrong second.
    detector = FakeDetector()
    analyzer = FrameAnalyzer(
        FakeModel(detections_per_frame=1), FakeTracker(), NullCollector(), detector
    )

    analyzer.analyze(_frame(), index=900)
    analyzer.analyze(_frame(), index=901)

    assert [frame_index for _, frame_index in detector.calls] == [900, 901]


def test_what_the_detector_reported_reaches_the_result():
    result = FrameAnalyzer(
        FakeModel(detections_per_frame=2), FakeTracker(), NullCollector(), FakeDetector()
    ).analyze(_frame(), index=5)

    assert [(v.track_id, v.frame_index) for v in result.violations] == [(1, 5), (2, 5)]


def test_an_analyzer_with_no_detector_reports_no_violations():
    # The unconfigured site: normal, not an error. Detection and tracking still run.
    result = FrameAnalyzer(FakeModel(detections_per_frame=2), FakeTracker()).analyze(
        _frame(), index=0
    )

    assert result.violations == []


def test_finishing_drains_whatever_the_rules_were_holding():
    # Empty for a rule, which decides on the frame it is given. A module working on a
    # clip is still holding a partial one when the frames run out, and the violation it
    # then reports names an earlier frame than the last one analysed.
    held = Violation(type="red_light_running", track_id=4, frame_index=880)
    analyzer = FrameAnalyzer(
        FakeModel(), FakeTracker(), NullCollector(), FakeDetector(held=[held])
    )

    analyzer.analyze(_frame(), index=900)

    assert analyzer.finish() == [held]


def test_an_analyzer_with_no_detector_holds_nothing_back():
    assert FrameAnalyzer(FakeModel(), FakeTracker()).finish() == []


def test_a_job_with_no_configuration_gets_a_detector_with_no_rules():
    # A site nobody has annotated yet. Detection and tracking still run; there is
    # simply nothing to judge them against.
    analyzer = make_analyzer(FakeModel(), fps=30.0, new_tracker=_trackers())

    assert analyzer._detector.get_modules() == ()


def test_a_job_with_a_configuration_gets_a_detector_built_from_it():
    analyzer = make_analyzer(
        FakeModel(),
        fps=30.0,
        configuration=CONFIGURATION,
        types=["red_light_running"],
        new_tracker=_trackers(),
    )

    assert [module.type for module in analyzer._detector.get_modules()] == [
        "red_light_running"
    ]


def test_the_detector_is_told_which_violations_the_job_asked_for():
    # Canonical values, so the worker holds no table mapping its ViolationType to the
    # names a document uses.
    built: list[tuple] = []

    make_analyzer(
        FakeModel(),
        fps=25.0,
        configuration=CONFIGURATION,
        types=["pedestrian_right_of_way"],
        new_tracker=_trackers(),
        new_detector=lambda configuration, types, fps: built.append((types, fps))
        or Detector(),
    )

    assert built == [(["pedestrian_right_of_way"], 25.0)]


def test_the_detector_is_given_the_same_frame_rate_as_the_tracker():
    # A source ffprobe could not read. A module reasoning about a span of time has to
    # agree with the tracker about how much time a frame is worth.
    trackers = _trackers()
    built: list[float] = []

    make_analyzer(
        FakeModel(),
        fps=None,
        configuration=CONFIGURATION,
        new_tracker=trackers,
        new_detector=lambda configuration, types, fps: built.append(fps) or Detector(),
    )

    assert trackers.made[0].fps == built[0] == DEFAULT_FPS


def test_a_configuration_that_cannot_be_parsed_stops_the_job():
    # Before a frame is decoded, and loudly — the same bargain a bad calibration
    # strikes. A job that watched an hour of footage through the wrong polygons would
    # look entirely normal and be entirely wrong.
    with pytest.raises(ConfigurationInvalid):
        make_analyzer(
            FakeModel(),
            fps=30.0,
            configuration={"version": 99, "violations": ["rlr_violation"], "regions": {}},
            new_tracker=_trackers(),
        )
