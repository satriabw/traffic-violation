import numpy as np
import pytest

from trajectory_collector import CameraModel, CollectorParams, PinholeCollector
from trajectory_collector.pinhole import anchor_points

# A camera looking straight down from 100m with a 1000px focal length: the ground is
# the image scaled by 0.1 metres per pixel, so every expectation below is arithmetic
# anyone can check. At 10fps, a box moving 10 px per frame is moving 10 m/s.
METRES_PER_PIXEL = 0.1
FPS = 10.0
SPEED = 10.0
PIXELS_PER_FRAME = SPEED / FPS / METRES_PER_PIXEL


def _camera() -> CameraModel:
    return CameraModel.from_matrices(
        camera_matrix=[[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
        rot_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        tvec=[0.0, 0.0, 100.0],
    )


def _collector(params: CollectorParams = CollectorParams()) -> PinholeCollector:
    return PinholeCollector(_camera(), fps=FPS, params=params)


def _boxes(*bottom_centres: tuple[float, float]) -> np.ndarray:
    """Boxes whose bottom edges are centred on the given points."""
    return np.array(
        [[x - 50.0, y - 80.0, x + 50.0, y] for x, y in bottom_centres], dtype=float
    )


def _moving(frame: int) -> np.ndarray:
    """One box, travelling along x at SPEED, at the frame's true position."""
    return _boxes((500.0 + PIXELS_PER_FRAME * frame, 500.0))


def _drive(collector: PinholeCollector, frames: range, ids=(1,)):
    """Run a straight-line track through the collector, returning the last reading."""
    latest = {}
    for frame in frames:
        latest = collector.collect(_moving(frame), np.array(ids), frame)
    return latest


# --- the anchor ---------------------------------------------------------------


def test_a_box_is_anchored_where_it_meets_the_ground():
    # The bottom edge, not the centre. A car's box centre floats about a metre above
    # the road, and projecting a point that is not on the ground plane onto the ground
    # plane puts it metres from the car.
    points = anchor_points(np.array([[100.0, 200.0, 300.0, 400.0]]))

    assert points == pytest.approx(np.array([[200.0, 400.0]]))


# --- warming up ---------------------------------------------------------------


def test_a_track_reports_no_speed_until_it_has_been_seen_enough():
    # Not because it is stationary, but because nothing yet knows. A speed differenced
    # from two noisy sightings would be worse than no speed at all.
    collector = _collector()

    first = collector.collect(_moving(0), np.array([1]), 0)
    second = collector.collect(_moving(1), np.array([1]), 1)

    assert first[1].speed == 0.0
    assert second[1].speed == 0.0


def test_a_track_reports_its_projected_position_while_warming_up():
    # No speed yet, but the position is known from the first sighting — 500 px at
    # 0.1 m/px is 50 m, and reporting nothing at all would lose a frame of evidence.
    collector = _collector()

    first = collector.collect(_moving(0), np.array([1]), 0)

    assert first[1].position == pytest.approx((50.0, 50.0))


def test_a_speed_arrives_once_the_warmup_is_done():
    collector = _collector()

    last = _drive(collector, range(CollectorParams().warmup_frames))

    assert last[1].speed > 0.0


def test_how_long_the_warmup_lasts_is_tunable():
    collector = _collector(CollectorParams(warmup_frames=5))

    assert _drive(collector, range(4))[1].speed == 0.0
    assert _drive(collector, range(4, 5))[1].speed > 0.0


# --- speed --------------------------------------------------------------------


def test_a_track_at_a_known_speed_reports_it():
    last = _drive(_collector(), range(30))

    assert last[1].speed == pytest.approx(SPEED, abs=0.1)


def test_a_track_at_a_known_position_reports_it():
    # 30 frames at 10 px per frame from 500 px, at 0.1 m/px: 80 metres along.
    last = _drive(_collector(), range(31))

    assert last[1].position[0] == pytest.approx(80.0, abs=0.1)


def test_a_stationary_object_is_reported_as_stationary():
    collector = _collector()

    latest = {}
    for frame in range(30):
        latest = collector.collect(_moving(0), np.array([1]), frame)

    assert latest[1].speed == pytest.approx(0.0, abs=0.1)


# --- frame indices are absolute -----------------------------------------------


def test_the_same_track_reads_the_same_wherever_the_chunk_starts():
    # The reason collect takes an absolute index. A worker processing frames 900-1800
    # of a video is watching the same motion as one processing 0-900, and must measure
    # the same speed from it.
    from_zero = _collector()
    from_nine_hundred = _collector()

    for frame in range(30):
        from_zero.collect(_moving(frame), np.array([1]), frame)
        from_nine_hundred.collect(_moving(frame), np.array([1]), 900 + frame)

    assert from_zero.collect(_moving(30), np.array([1]), 30)[1].speed == pytest.approx(
        from_nine_hundred.collect(_moving(30), np.array([1]), 930)[1].speed
    )


def test_a_track_that_vanishes_and_returns_is_measured_over_the_real_interval():
    # Seen for six frames, gone for thirty, then back three seconds further down the
    # road. Measured against elapsed time it is still doing 10 m/s; measured as though
    # only one frame had passed it would look like 300.
    collector = _collector()
    _drive(collector, range(6))

    after = collector.collect(_moving(36), np.array([1]), 36)

    assert after[1].position[0] == pytest.approx(50.0 + 36.0, abs=1.0)
    assert after[1].speed == pytest.approx(SPEED, abs=2.0)


def test_a_track_sampled_every_few_frames_still_reports_its_speed():
    # Not every caller runs detection on every frame. Gaps in the index are elapsed
    # time, so a track sampled every third frame moves three frames' worth each time.
    collector = _collector()

    latest = {}
    for frame in range(0, 90, 3):
        latest = collector.collect(_moving(frame), np.array([1]), frame)

    assert latest[1].speed == pytest.approx(SPEED, abs=0.5)


# --- several tracks -----------------------------------------------------------


def test_two_tracks_are_filtered_independently():
    collector = _collector()

    latest = {}
    for frame in range(30):
        boxes = _boxes(
            (500.0 + PIXELS_PER_FRAME * frame, 500.0),  # moving
            (200.0, 300.0),  # parked
        )
        latest = collector.collect(boxes, np.array([1, 2]), frame)

    assert latest[1].speed == pytest.approx(SPEED, abs=0.1)
    assert latest[2].speed == pytest.approx(0.0, abs=0.1)


def test_only_what_is_in_this_frame_is_reported():
    # A track that has gone is absent rather than repeated, so a caller can tell
    # "still here" from "was here once".
    collector = _collector()

    for frame in range(5):
        collector.collect(_boxes((500.0, 500.0), (200.0, 300.0)), np.array([1, 2]), frame)
    latest = collector.collect(_boxes((500.0, 500.0)), np.array([1]), 5)

    assert set(latest) == {1}


def test_an_empty_frame_reports_nothing():
    collector = _collector()

    latest = collector.collect(
        np.empty((0, 4)), np.empty(0, dtype=int), frame_index=0
    )

    assert latest == {}


# --- what cannot be located ---------------------------------------------------


def test_a_box_that_does_not_meet_the_ground_is_skipped_not_reported():
    # Above the horizon there is no ground point. Letting the resulting nan into a
    # Kalman filter would not cost one frame — it would poison every estimate that
    # track produced afterwards.
    tilted = CameraModel.from_matrices(
        camera_matrix=[[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
        rot_matrix=[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        tvec=[0.0, 5.0, 40.0],
    )
    collector = PinholeCollector(tilted, fps=FPS)

    latest = collector.collect(_boxes((0.0, 500.0), (0.0, 0.0)), np.array([1, 2]), 0)

    assert set(latest) == {1}


def test_a_frame_rate_of_zero_is_refused():
    # Every interval the filter works in is derived from it, so this is not a degraded
    # mode — it is a division by zero waiting for the first frame.
    with pytest.raises(ValueError, match="fps must be positive"):
        PinholeCollector(_camera(), fps=0.0)
