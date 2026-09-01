"""What the record says, worked out here so the model never has to.

THE MODEL IS NEVER ASKED TO COUNT. Everything in this module is arithmetic over the track
record that the prompt hands over as a finished statement — who was present when the rule
fired, which speeds are impossible, what never moved. That is not a stylistic preference:
a three-arm study over this exact data (`research/violation-prompt-study/round3/`) asked
two independent arms to screen twenty-six tracks for implausible speeds; one answered six,
the other seven, and the true figure is nine. A miscount lands in a document supporting
enforcement against a real person, and nothing downstream can catch it.

MOVEMENT IS MEASURED ON THE GROUND, IN METRES, off `trajectory`. Pixels needed no
calibration and that was their whole appeal, but they answer about the image rather than
the road: a car receding down its lane crosses very few of them, and a near object
shuffling in place crosses many. The projection is an estimate and this camera's is a poor
one, but it is an estimate of the thing being asked about. Where there is no ground plane
the answer is None and the caller says so — never a distance measured somewhere else,
which is the hundred-fold error `TrackSummary.trajectory` warns about arriving by a
different door.
"""

from shared.models.violation import TrackSummary, ViolationMetadata

from llm_service.constant import IMPLAUSIBLE_MPS


def seconds(frame_index: int | None, fps: float | None) -> float | None:
    """A frame index as a time into the clip, or None if nothing can convert it."""
    if frame_index is None or not fps or fps <= 0:
        return None
    return frame_index / fps


def moment(frame_index: int | None, fps: float | None) -> str:
    """Where in the footage this happened, in the terms somebody scrubbing it would use.

    Falls back to naming no position at all rather than to the frame index. A source
    nobody probed has no frame rate, and a clerk handed "frame 159" has been given a
    number they cannot act on and would have to ask an engineer to interpret.
    """
    at = seconds(frame_index, fps)
    if at is None:
        return "at the moment the vehicle was flagged"
    return f"{at:.1f} seconds into the footage"


def travel(track: TrackSummary) -> float | None:
    """How far this object moved across its whole life, on the ground, in metres.

    Manhattan against the first fix — |dx| + |dy| — rather than straight-line. It costs a
    subtraction and no square root, and it can only overstate, by at most half again on a
    perfect diagonal, which is inside the margin `STATIC_METRES` carries anyway.

    Deliberately not a speed: it answers "did this thing go anywhere", which one broken
    frame cannot swing the way it swings a derivative.

    None, never 0.0, when nothing was projected — a job with no calibration has no ground
    plane, so every entry in `trajectory` is None. "Did not move" and "was never measured"
    are different findings and the caller has to say different things about them.
    """
    fixes = [point for point in track.trajectory if point is not None]
    if len(fixes) < 2:
        return None
    first = fixes[0]
    return max(abs(x - first[0]) + abs(y - first[1]) for x, y in fixes)


def present(track: TrackSummary, frame_index: int | None) -> bool:
    """Was this object there at the moment the rule fired?

    MEMBERSHIP, NOT SPAN. Asking whether the violation frame falls between a track's first
    and last sample is a different and much weaker question, and on this data it is wrong
    often enough to matter: the evidence window ends on the violation frame, so every
    object still on screen when the buffer closed passes that test — seventeen of
    twenty-six in the scene this was measured on. Tracks also have gaps.
    """
    return frame_index is not None and frame_index in track.frame_idxs


def implausible(track: TrackSummary) -> float | None:
    """This track's top speed, if it is one nothing on a road could have done."""
    speeds = [speed for speed in track.speed if speed is not None]
    if not speeds or max(speeds) <= IMPLAUSIBLE_MPS:
        return None
    return max(speeds)


def violator(metadata: ViolationMetadata | None) -> TrackSummary | None:
    """The track the detector convicted, if it recorded which one.

    No falling back to the first vehicle. That guess was safe only while a violation's
    record held exactly one of them; the record is now the whole scene, and picking the
    first would silently name the car queued behind the offender.
    """
    if metadata is None or metadata.violator_track_id is None:
        return None
    for track in metadata.vehicles:
        if track.track_id == metadata.violator_track_id:
            return track
    return None
