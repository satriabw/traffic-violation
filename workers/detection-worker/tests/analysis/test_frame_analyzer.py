import numpy as np
import supervision as sv

from detection_worker.analysis.frame_analyzer import FrameAnalyzer, make_analyzer


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
    """Records every update, and numbers whatever it is given 1..n."""

    def __init__(self, fps=None):
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
