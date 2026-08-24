"""What one frame produced.

A record rather than a tuple because it grew: detections, then trajectories, now what
the rules decided. Every one of those is a field here rather than another return value
the handler has to keep in the right order.

The frame itself is deliberately not a field. Nothing downstream of the analyzer needs
the pixels yet — evidence frames are cropped where a rule fires, inside the analyzer,
not by whoever reads this — and holding a reference to every decoded frame is how a
long chunk turns into a memory problem.
"""

from dataclasses import dataclass, field

import supervision as sv
from trajectory_collector import Trajectory
from violation_detector import Violation


@dataclass(frozen=True)
class FrameResult:
    # Absolute — the frame's position in the video, not its position in this run. A job
    # covering frames 900-1800 reports 900 for its first frame, because that is the
    # number a violation has to be recorded against for anyone to find it in the
    # footage later.
    index: int
    # Tracked: these have been through the tracker, so they carry `tracker_id`.
    detections: sv.Detections
    # Where each tracked object is on the ground and how fast it is going, keyed by the
    # same tracker id. Empty for a site with no calibration, which is a normal state
    # rather than a failure: without one there is no ground plane to project onto.
    # Keyed rather than aligned with `detections` because that is the join a rule
    # actually makes — "how fast is track 7" — and an index into a per-frame array
    # would mean something different on every frame.
    trajectories: dict[int, Trajectory]
    # What the rules saw on this frame, which on almost every frame is nothing. A
    # default, unlike the two above, because a frame with no violations is the normal
    # case and a caller assembling one by hand should not have to say so.
    #
    # A violation's own `frame_index` is not necessarily this frame's `index`: a rule
    # reports on the frame it was given, but a module working on a clip reports several
    # frames late. Record the violation's, never the result's.
    violations: list[Violation] = field(default_factory=list)

    @property
    def track_ids(self) -> list[int]:
        """The tracker ids on this frame, or none at all.

        `tracker_id` is None rather than an empty array on a frame the tracker had
        nothing to assign, which is most frames of most footage.
        """
        if self.detections.tracker_id is None:
            return []
        return [int(tracker_id) for tracker_id in self.detections.tracker_id]
