"""What one frame produced.

A record rather than a tuple because it is about to grow: trajectories hang off the
same frame, and so do whatever the rule engine decides about it. Every one of those is
a field here rather than another return value the handler has to keep in the right
order.

The frame itself is deliberately not a field. Nothing downstream of the analyzer needs
the pixels yet — evidence frames are cropped where a rule fires, inside the analyzer,
not by whoever reads this — and holding a reference to every decoded frame is how a
long chunk turns into a memory problem.
"""

from dataclasses import dataclass

import supervision as sv


@dataclass(frozen=True)
class FrameResult:
    # Absolute — the frame's position in the video, not its position in this run. A job
    # covering frames 900-1800 reports 900 for its first frame, because that is the
    # number a violation has to be recorded against for anyone to find it in the
    # footage later.
    index: int
    # Tracked: these have been through the tracker, so they carry `tracker_id`.
    detections: sv.Detections

    @property
    def track_ids(self) -> list[int]:
        """The tracker ids on this frame, or none at all.

        `tracker_id` is None rather than an empty array on a frame the tracker had
        nothing to assign, which is most frames of most footage.
        """
        if self.detections.tracker_id is None:
            return []
        return [int(tracker_id) for tracker_id in self.detections.tracker_id]
