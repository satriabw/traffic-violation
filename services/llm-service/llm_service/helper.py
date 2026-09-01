"""What the record says, worked out here so the model never has to.

THE MODEL IS NEVER ASKED TO COUNT. Everything in this module is arithmetic over the track
record that the prompt hands over as a finished statement — who was present when the rule
fired, which speeds are impossible, what never moved. That is not a stylistic preference:
a three-arm study over this exact data (`research/violation-prompt-study/round3/`) asked
two independent arms to screen twenty-six tracks for implausible speeds; one answered six,
the other seven, and the true figure is nine. A miscount lands in a document supporting
enforcement against a real person, and nothing downstream can catch it.

EVERYTHING HERE READS `bboxes` AND `frame_idxs`, NEVER `trajectory` OR `speed`, except
where the point is to catch those two failing. Image-space pixels need no calibration, so
they survive one being absent — `TrackSummary.trajectory` is entirely None on a job with
no calibration — and they survive one being broken, which on this camera it is.
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


def travel(track: TrackSummary) -> float:
    """How far this object's centre moved across its whole life, in the image.

    Deliberately not a speed and deliberately not in metres: it answers "did this thing
    move at all", which needs no calibration and survives one being broken.
    """
    boxes = track.bboxes
    if len(boxes) < 2:
        return 0.0
    first = ((boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2)
    return max(
        (((box[0] + box[2]) / 2 - first[0]) ** 2 + ((box[1] + box[3]) / 2 - first[1]) ** 2)
        ** 0.5
        for box in boxes
    )


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
