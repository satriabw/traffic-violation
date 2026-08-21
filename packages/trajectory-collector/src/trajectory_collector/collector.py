"""What a trajectory collector is, and what it produces.

A collector turns pixel boxes into positions and speeds on the ground plane. The
projection it uses comes from a camera calibration, so a collector is built from one —
`TrajectoryCollector.from_calibration(...)` — and nothing that uses a collector ever
has to know which projection it got.

That indirection is the whole reason this is a package. The caller holds boxes and
track ids; it should not also hold a camera model, a Kalman filter and the knowledge of
how to wire them together, any more than a caller of a video library holds a decoder.

UNITS ARE METRES, throughout, and there is no field to say otherwise. A calibration is
built in some real-world unit and every number downstream inherits it, so declaring the
unit per document would only move the problem — a speed threshold written against
metres and evaluated against a calibration built in feet is wrong either way. Metres is
the contract; a calibration in anything else is a wrong calibration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Where one tracked object is, and how fast it is going, right now.

    The latest state, not a history. A track's past positions are the collector's
    business — it needs them to filter — and keeping them here would mean every caller
    holding a growing list it almost certainly only reads the end of.
    """

    # On the ground plane, in metres. Not pixels: two objects the same distance apart
    # on screen are metres apart at the bottom of a frame and tens of metres apart at
    # the top, which is exactly what projecting fixes.
    position: tuple[float, float]
    # Metres per second, on the ground plane. A magnitude — heading is recoverable from
    # successive positions, and nothing so far has wanted it.
    speed: float


class TrajectoryCollector(ABC):
    """Boxes in, ground-plane trajectories out, one frame at a time.

    Stateful, and single-use per video: a collector holds a filter per track, keyed by
    tracker id. Ids restart at 1 for every tracking session, so feeding two videos
    through one collector would silently merge unrelated objects.
    """

    @classmethod
    def from_calibration(
        cls,
        source: str | PathLike | bytes | bytearray,
        fps: float,
    ) -> "TrajectoryCollector":
        """Build the collector a calibration calls for.

        `source` is either a path to an OpenCV FileStorage calibration or the
        document's own bytes — a caller that read its calibration from object storage
        has the document in hand and should not have to write it to a temp file to be
        allowed to use it.

        Which collector comes back is this function's decision, taken from what the
        document contains. That is the point: callers name one class.

        Imported here rather than at module scope because the collector it builds
        subclasses this one, and a base class that imports its own subclasses at import
        time cannot be imported at all.
        """
        from trajectory_collector.pinhole import collector_from_calibration

        return collector_from_calibration(source, fps)

    @abstractmethod
    def collect(
        self,
        boxes: np.ndarray,
        track_ids: np.ndarray,
        frame_index: int,
    ) -> dict[int, Trajectory]:
        """The trajectory of everything visible in this frame, keyed by track id.

        `boxes` is (N, 4) in pixels, `xyxy`. `track_ids` is (N,) of ints, aligned with
        it row for row — an object's box is `boxes[i]` exactly when its id is
        `track_ids[i]`. Arrays rather than a list of objects because the projection is
        one matrix multiply over all of them, and because arrays are the one shape
        every detection library already has.

        `frame_index` is ABSOLUTE — the frame's position in the video, not the number
        of times this method has been called. A collector filtering over time needs to
        know how long a track was missing for, and a caller processing a chunk starting
        at frame 900, or sampling every third frame, would otherwise be measuring gaps
        against a counter that has nothing to do with elapsed time.

        Only what is visible in this frame comes back. A track that has gone is absent
        rather than repeated, so a caller can tell "still here" from "was here once".
        """


class NullCollector(TrajectoryCollector):
    """A collector for a video with no calibration behind it.

    Without a calibration there is no ground plane, so there is no honest position to
    report — and a video with no calibration is a normal state, not an error. This says
    "nothing to report" in the shape everything downstream already handles, so the
    caller keeps one code path instead of a null check per frame.
    """

    def collect(
        self,
        boxes: np.ndarray,
        track_ids: np.ndarray,
        frame_index: int,
    ) -> dict[int, Trajectory]:
        return {}
