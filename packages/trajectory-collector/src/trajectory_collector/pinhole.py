"""The collector: boxes to filtered positions and speeds on the ground.

Three steps per frame, in this order and for these reasons.

**Anchor.** A box becomes one point: the middle of its bottom edge, where the object
meets the ground. Not its centre — the centre of a car's box floats about a metre above
the road, and projecting a point that is not on the ground plane onto the ground plane
puts it metres away from the car, further the shallower the camera angle. The bottom
edge is the only anchor the Z=0 assumption is actually true for.

**Project.** Every anchor in the frame at once, through the camera model. One matrix
multiply, because per-object projection would be the same arithmetic done N times.

**Filter.** Each track's projected position goes through its own Kalman filter, which
is where speed comes from. Speed is never differenced from consecutive positions: a
detection box jitters by a few pixels, near the horizon a few pixels is metres, and the
resulting frame-to-frame speed swings wildly while the object moves smoothly.

A collector holds one filter per tracker id, so it belongs to one tracking session. Ids
restart at 1 for every session, so a collector outliving one would feed a new object's
measurements into a departed object's filter.
"""

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Mapping

import numpy as np

from trajectory_collector.camera_model import CameraModel
from trajectory_collector.collector import Trajectory, TrajectoryCollector
from trajectory_collector.kalman import POSITION, VELOCITY, FilterParams, KalmanFilter


@dataclass(frozen=True)
class CollectorParams:
    """Tuning, shared by every track. Frozen so one track cannot alter the next's."""

    # How many sightings before a track's filter starts. A filter needs a velocity to
    # start from, and one frame gives no velocity at all — so the first few sightings
    # are spent measuring one. Raising this buys a better initial velocity and delays
    # the first useful speed by the same amount.
    warmup_frames: int = 3
    # A track missing for longer than this comes back with its velocity and
    # acceleration uncertainty widened. One or two frames is ordinary flicker and the
    # model handles it; a longer absence means the old velocity has genuinely stopped
    # being evidence.
    gap_inflation_threshold: int = 2
    filter: FilterParams = FilterParams()


@dataclass
class _Track:
    """What the collector remembers about one tracker id.

    Only the filter and the last frame it was seen on — no position history. The
    filter's state *is* the summary of everything it has seen, and keeping the
    positions as well would mean a list per track growing for as long as the object
    stays in frame, to be read only at the end.
    """

    last_seen: int
    # Sightings from before the filter existed, with the frames they were seen on.
    # Cleared the moment it does.
    warmup: list[tuple[int, np.ndarray]] = field(default_factory=list)
    filter: KalmanFilter | None = None


def anchor_points(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) xyxy boxes to (N, 2) points where each object meets the ground."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))


class PinholeCollector(TrajectoryCollector):
    """Trajectories through a pinhole camera model. Built by `from_calibration`."""

    def __init__(
        self,
        camera_model: CameraModel,
        fps: float,
        params: CollectorParams = CollectorParams(),
    ):
        if fps <= 0:
            # Every interval this filter works in is derived from it, so a zero or
            # negative rate is not a degraded mode — it is a division by zero waiting
            # for the first frame.
            raise ValueError(f"fps must be positive, got {fps}")
        self._camera_model = camera_model
        self._params = params
        # The nominal interval between frames. Gaps are measured in multiples of it,
        # which is why frame indices have to be absolute.
        self._dt = 1.0 / fps
        self._tracks: dict[int, _Track] = {}

    def collect(
        self,
        boxes: np.ndarray,
        track_ids: np.ndarray,
        frame_index: int,
    ) -> dict[int, Trajectory]:
        if len(track_ids) == 0:
            return {}

        ground = self._camera_model.project_to_ground(anchor_points(boxes))
        # A box whose bottom edge lands on or above the horizon has no ground point,
        # and comes back non-finite. Such a track is skipped for this frame rather than
        # reported: there is no position to report, and letting a nan into a Kalman
        # filter would not cost one frame — it would poison every estimate that track
        # ever produces afterwards.
        on_the_ground = np.isfinite(ground).all(axis=1)
        return {
            int(track_id): self._advance(int(track_id), ground[row], frame_index)
            for row, track_id in enumerate(track_ids)
            if on_the_ground[row]
        }

    def _advance(
        self, track_id: int, position: np.ndarray, frame_index: int
    ) -> Trajectory:
        track = self._tracks.get(track_id)
        if track is None:
            track = self._tracks[track_id] = _Track(last_seen=frame_index)

        if track.filter is None:
            return self._warm_up(track, position, frame_index)

        frames_missing = frame_index - track.last_seen
        track.last_seen = frame_index

        if frames_missing > self._params.gap_inflation_threshold:
            # Scaled by the length of the gap: coming back after ten frames should
            # leave the filter more willing to be moved than coming back after three.
            track.filter.inflate_covariance(frames_missing)

        # The real elapsed interval, not one frame. A track seen on frames 10 and 40
        # moved for a second, and stepping the model by 1/30s would have it barely
        # move at all and then be dragged to the measurement as though it had teleported.
        track.filter.predict(dt=frames_missing * self._dt)
        return _trajectory(track.filter.correct(position))

    def _warm_up(
        self, track: _Track, position: np.ndarray, frame_index: int
    ) -> Trajectory:
        """Collect sightings until there are enough to start a filter from."""
        track.warmup.append((frame_index, position))
        track.last_seen = frame_index

        if len(track.warmup) < self._params.warmup_frames:
            # The projected position, unfiltered, and no speed — not because the object
            # is stationary but because nothing yet knows. Reporting a differenced
            # speed from two noisy sightings would be worse than reporting none.
            return Trajectory(position=_point(position), speed=0.0)

        (first_frame, first_position) = track.warmup[0]
        (last_frame, last_position) = track.warmup[-1]
        elapsed = (last_frame - first_frame) * self._dt
        # Measured over the frames the sightings actually span, not the number of them:
        # a track that flickered during warmup covered its displacement over a longer
        # interval and is therefore slower than the count would suggest.
        velocity = (last_position - first_position) / elapsed

        track.filter = KalmanFilter(
            x=first_position[0],
            y=first_position[1],
            vx=velocity[0],
            vy=velocity[1],
            dt=self._dt,
            params=self._params.filter,
        )

        # Replay the warmup through the filter it just seeded, so the state handed back
        # for this frame has seen every measurement rather than only the first.
        state = track.filter.x
        previous_frame = first_frame
        for seen_on, measurement in track.warmup[1:]:
            track.filter.predict(dt=(seen_on - previous_frame) * self._dt)
            state = track.filter.correct(measurement)
            previous_frame = seen_on

        track.warmup = []
        return _trajectory(state)


def _point(position: np.ndarray) -> tuple[float, float]:
    return (float(position[0]), float(position[1]))


def _trajectory(state: np.ndarray) -> Trajectory:
    """A filter state as the two numbers anyone outside this package wants."""
    # A magnitude, unrounded. How many decimals a speed deserves is a question about
    # displaying it, and the answer differs between a log line and a threshold
    # comparison — so the number that leaves here is the one the filter computed.
    return Trajectory(
        position=_point(state[POSITION].reshape(-1)),
        speed=float(np.linalg.norm(state[VELOCITY].reshape(-1))),
    )


def collector_from_calibration(
    source: str | PathLike | Mapping[str, Any], fps: float
) -> PinholeCollector:
    """The collector a calibration document calls for.

    One projection today, so this reads as a long way round to a constructor. It is the
    seam where a second one would be chosen — and having it here means adding one
    changes this function rather than every caller.
    """
    return PinholeCollector(CameraModel.from_calibration(source), fps)
