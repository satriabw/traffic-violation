import numpy as np
import supervision as sv

from detection_worker.detection.tracker import DEFAULT_FPS, DEFAULT_PARAMS, TrackerParams, make_tracker


class RecordingBackend:
    """Stands in for the tracking library so the construction arguments are visible.

    Everything about how a tracker is configured happens in its constructor, and no
    other seam exposes it.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates: list[sv.Detections] = []

    def update_with_detections(self, detections):
        self.updates.append(detections)
        return detections


def _recorder():
    made: list[RecordingBackend] = []

    def factory(**kwargs):
        backend = RecordingBackend(**kwargs)
        made.append(backend)
        return backend

    factory.made = made
    return factory


def _detection(x=10.0, y=10.0, confidence=0.9):
    return sv.Detections(
        xyxy=np.array([[x, y, x + 40, y + 40]], dtype=np.float32),
        confidence=np.array([confidence], dtype=np.float32),
        class_id=np.array([2], dtype=np.int16),
    )


# --- configuration ------------------------------------------------------------


def test_the_sources_frame_rate_reaches_the_backend():
    # frame_rate scales the lost-track buffer into real time, so a tracker built for
    # 25fps footage holds a lost track for a different number of frames than one built
    # for 60fps. This is the reason a tracker cannot be shared between jobs.
    factory = _recorder()

    make_tracker(fps=25.0, backend_factory=factory)

    assert factory.made[0].kwargs["frame_rate"] == 25.0


def test_an_unknown_frame_rate_falls_back_to_a_sane_default():
    # JobSource.fps is optional — a source whose probe could not read a rate still has
    # to be trackable.
    factory = _recorder()

    make_tracker(fps=None, backend_factory=factory)

    assert factory.made[0].kwargs["frame_rate"] == DEFAULT_FPS


def test_a_zero_frame_rate_falls_back_too():
    # ffprobe reports unknown rates as 0/0, which arrives here as 0.0 and would make
    # the lost-track buffer meaningless.
    factory = _recorder()

    make_tracker(fps=0.0, backend_factory=factory)

    assert factory.made[0].kwargs["frame_rate"] == DEFAULT_FPS


def test_the_tuning_parameters_reach_the_backend():
    factory = _recorder()
    params = TrackerParams(track_activation_threshold=0.4, lost_track_buffer=90)

    make_tracker(fps=30.0, params=params, backend_factory=factory)

    kwargs = factory.made[0].kwargs
    assert kwargs["track_activation_threshold"] == 0.4
    assert kwargs["lost_track_buffer"] == 90
    assert kwargs["minimum_matching_threshold"] == DEFAULT_PARAMS.minimum_matching_threshold


def test_update_passes_detections_straight_through_to_the_backend():
    factory = _recorder()
    tracker = make_tracker(fps=30.0, backend_factory=factory)
    detections = _detection()

    tracker.update(detections)

    assert factory.made[0].updates == [detections]


# --- against the real tracking library ----------------------------------------


def test_tracked_detections_come_back_with_ids():
    tracker = make_tracker(fps=30.0)

    tracked = tracker.update(_detection())

    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 1


def test_an_object_keeps_its_id_across_frames():
    # The whole point of tracking: the rule engine keys its per-vehicle state on this
    # id, so an id that changed frame to frame would reset every vehicle's history.
    tracker = make_tracker(fps=30.0)

    ids = [tracker.update(_detection(x=10.0 + step)).tracker_id[0] for step in range(5)]

    assert len(set(ids)) == 1


def test_two_trackers_do_not_share_state():
    # The property the per-job lifetime exists to guarantee. If these shared state, a
    # track from one site's chunk could be re-matched against another's.
    first, second = make_tracker(fps=30.0), make_tracker(fps=30.0)

    for _ in range(3):
        first.update(_detection())
    fresh = second.update(_detection()).tracker_id[0]

    # Both trackers number their own tracks from 1, independently. Which also means
    # ids DO collide across chunks — two unrelated vehicles in adjacent chunks are
    # both "1" — so overlap dedup has to match on position, never on id.
    assert fresh == 1


def test_a_frame_with_no_detections_is_still_a_frame():
    # Skipping the update on an empty frame would let the tracker's frame counter fall
    # behind the video, and every lost-track age with it.
    tracker = make_tracker(fps=30.0)

    tracked = tracker.update(sv.Detections.empty())

    assert len(tracked) == 0
