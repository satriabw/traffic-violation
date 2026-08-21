"""Multi-object tracking: detections in, the same detections carrying stable ids out.

A thin wrapper over supervision's ByteTrack, and the thinness is the point — the value
here is not the code but the lifetime it enforces. A tracker holds live state (active
tracks, lost tracks, a Kalman filter per track, a frame counter), so it belongs to
exactly one job. Sharing one across jobs would let a track from one site's chunk be
re-matched against another site's, and that corruption would arrive silently in the
id-keyed caches the rule engine keeps.

The tuning, by contrast, is the same everywhere and lives in TrackerParams at module
scope. That split — shared configuration, per-job state — is what people usually reach
for a singleton to get, and it gets it without the shared state.

ON THE BACKEND: `sv.ByteTrack` is deprecated as of supervision 0.28 and is removed in
0.31, which is why the dependency is pinned below it. Upstream points at
`ByteTrackTracker` in the separate `trackers` package, but that package depends on
`opencv-python` where this worker deliberately runs `opencv-python-headless`, and two
OpenCV builds in one environment is a worse problem than a pinned version. When that
changes, this module is the only thing that has to: construct the other backend and
call `update()` instead of `update_with_detections()`.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable

import supervision as sv

# What ffprobe could not tell us. Any value would be arbitrary; this one at least
# matches the rate most traffic footage is shot at, and only affects how long a lost
# track survives before the tracker forgets it.
DEFAULT_FPS = 30.0


def resolve_fps(fps: float | None) -> float:
    """The frame rate to work in, given whatever the source could tell us.

    One definition, because more than one thing scales by it. The tracker ages lost
    tracks in frames and the trajectory collector measures gaps in seconds, so a job
    whose tracker fell back to 30 while its collector fell back to something else would
    disagree with itself about how much time a frame is worth.
    """
    return fps if fps and fps > 0 else DEFAULT_FPS


@dataclass(frozen=True)
class TrackerParams:
    """Tuning, shared by every job. Frozen so one job cannot alter the next one's."""

    # Below this confidence a detection cannot start a new track. It can still be
    # matched to an existing one, which is what lets ByteTrack hold a track through a
    # partial occlusion.
    track_activation_threshold: float = 0.25
    # How many frames a lost track is kept alive for re-matching. Scaled by frame_rate
    # into real time, which is why the source's fps has to reach the constructor.
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    # 1 means a track is valid from its first frame. Raising it suppresses tracks born
    # from one-frame false positives, at the cost of missing genuinely brief ones.
    minimum_consecutive_frames: int = 1


DEFAULT_PARAMS = TrackerParams()


class Tracker:
    """Detections in, detections with `tracker_id` out."""

    def __init__(self, backend: Any):
        self._backend = backend

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Advance the tracker by one frame.

        Call this for every frame, including frames with nothing in them: the backend
        counts frames by counting updates, so a skipped empty frame would make every
        lost track appear younger than it is.
        """
        return self._backend.update_with_detections(detections)


def make_tracker(
    fps: float | None,
    params: TrackerParams = DEFAULT_PARAMS,
    backend_factory: Callable[..., Any] = sv.ByteTrack,
) -> Tracker:
    """A tracker for one job. Cheap to build — it allocates a few empty lists — so
    there is nothing to gain by reusing one, and correctness to lose."""
    return Tracker(
        backend_factory(
            frame_rate=resolve_fps(fps),
            **asdict(params),
        )
    )
