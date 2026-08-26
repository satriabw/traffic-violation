"""What one frame is worth remembering about, and what a caller hands in.

The whole vocabulary this package has. An object on a frame, and the frame it was on
— no detector, no tracker, no camera model, and no opinion about what any of it means.

NOTHING HERE IS INTERPRETED. `class_name` is carried and never compared to anything:
which names are vehicles and which are pedestrians is a question about traffic rules,
and it is already answered in one place elsewhere. A second copy of that vocabulary in
here would be a copy to keep in step, so this package holds the label and lets whoever
reads a window decide what it means.

FROZEN, AND MEANT IT. A record kept here is read minutes of footage later, when the
frame it describes is long gone and nothing can check it. The pipeline this is ported
from wrote its evidence as dicts and then, at save time, looped over the ones still
sitting in its buffer assigning `ann['speed']` and `ann['location']` into them — so
the buffer's contents changed as a side effect of reading it. Values here cannot be
edited after the fact, and a list handed in becomes a tuple on the way past, so a
caller that keeps hold of what it passed cannot alter what was recorded.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectState:
    """One tracked object, on one frame.

    Built from plain numbers rather than from any library's detection type — the same
    constraint the rest of this repository's packages hold, and what lets the caller's
    detector, tracker and camera model all be somebody else's problem.
    """

    # The tracker's id. Stable across the frames of one tracking session and
    # meaningless outside it: ids restart at 1 for every video, so a window is only
    # ever readable alongside the run that produced it.
    track_id: int
    # (x1, y1, x2, y2) in pixels, top-left origin — the convention supervision and
    # every detection library in reach already use.
    bbox: tuple[float, float, float, float]
    # Whatever the detector called it, carried verbatim and never compared to
    # anything here. See the module docstring.
    class_name: str
    # On the ground plane, in metres, or None where the job had no calibration to
    # project with. None rather than the pixel anchor as a stand-in: an uncalibrated
    # run is a normal state, and a number in this field has to mean metres on the
    # ground or it means nothing at all. Whoever reads the window decides what to
    # show when it is absent.
    position: tuple[float, float] | None = None
    # Metres per second on the ground plane, or None on the same terms. Absent for a
    # track's first few frames even under a calibration — a filter needs a velocity to
    # start from, and one sighting gives none.
    speed: float | None = None


@dataclass(frozen=True)
class FrameEntry:
    """Everything one frame contributed to the record.

    One of these per frame analysed, whether or not anything interesting was on it.
    The frames with nothing on them are half of what makes a window worth reading: a
    crossing that was empty for four seconds and then was not is a different story
    from one nobody ever looked at.
    """

    # ABSOLUTE — the frame's position in the video, not a count of how many have been
    # recorded. It is the only thing that can find this moment in the footage again,
    # which is the entire premise of keeping records instead of pixels.
    frame_index: int
    objects: tuple[ObjectState, ...] = ()
    # The caller's clock, and deliberately not one read in here. `time.time()` at the
    # moment of recording is when the *analysis* ran, which for a video file is hours
    # or days after the thing it describes and is nobody's idea of when a violation
    # happened. A caller reading a file passes footage time; a caller reading a live
    # stream passes the wall clock, and both are right. None where there is no clock
    # worth quoting — a source that never declared when it was recorded.
    timestamp: float | None = None

    def __post_init__(self):
        # The one coercion in this package, and the module docstring says why: a list
        # kept by the caller stays editable by the caller, and the pipeline this is
        # ported from edited exactly that. Frozen stops the field being rebound, not
        # the list behind it being appended to.
        if not isinstance(self.objects, tuple):
            object.__setattr__(self, "objects", tuple(self.objects))

    def track_ids(self) -> frozenset[int]:
        """Which tracks this frame saw. Frozen, for the same reason everything is."""
        return frozenset(state.track_id for state in self.objects)

