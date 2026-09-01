"""The paragraphs the note is assembled from, one function per section.

Each takes the request and returns finished prose for the clerk. Everything countable in
them was counted in `llm_service.helper`; everything they turn on is in
`llm_service.constant`. What is left here is the wording — which is the part that has to
change when a round of the study says the model read something the wrong way, and the
reason it is worth having on its own away from the arithmetic.
"""

from shared.models.detection import ViolationType
from shared.models.explanation import ExplainRequest
from shared.models.violation import EvidenceStatus

from llm_service import helper
from llm_service.constant import (
    BRIEF_SECONDS,
    PLATE_MARGINAL_PX,
    PLATE_RATIO,
    PLATE_READABLE_PX,
    STATIC_PX,
)


def speed_finding(request: ExplainRequest) -> str:
    """Whether any speed on this record can be used, and the reason it cannot.

    THE TWO REASONS ARE DIFFERENT AND THE ANSWER IS THE SAME. A violation judged with no
    calibration pinned has no pixel-to-world mapping at all; one judged with a calibration
    that is producing hundreds of km/h has a mapping that ran and failed. The earlier
    rounds only guarded the first, which turned out to be exactly backwards for the live
    data — the calibrated violation is the one carrying 331,529 km/h, and the prompt was
    vouching for it.

    So the check runs whatever the calibration says, and it runs across every object
    rather than the flagged one alone. That distinction is the finding: one impossible
    number is a bad reading on one vehicle, and nine of them is a broken camera.
    """
    tracks = request.metadata.vehicles if request.metadata else []
    failures = [speed for speed in (helper.implausible(track) for track in tracks) if speed]

    if request.calibration_id is None:
        return (
            "SPEED AND DISTANCE: unusable, and no figure derived from them may appear "
            "anywhere in your note.\n"
            "  This footage was recorded with no camera calibration set up, so the system "
            "has no way to turn what it saw into distances. There is nothing to "
            "sanity-check — the numbers were never anchored to anything.\n"
            "  Tell the clerk plainly to disregard any speed or distance shown against "
            "this violation."
        )

    if failures:
        worst = max(failures) * 3.6
        many = len(failures) > 1
        return (
            "SPEED AND DISTANCE: unusable, and no figure derived from them may appear "
            "anywhere in your note.\n"
            f"  A calibration was set up for this camera, but it is producing figures "
            f"that are not physically possible: {len(failures)} of {len(tracks)} objects "
            f"tracked in this clip register speeds up to {worst:,.0f} km/h.\n"
            + (
                "  That many failures across unrelated vehicles means the camera's "
                "distance calibration is faulty. It is not several drivers speeding, and "
                "it should be reported as an equipment problem.\n"
                if many
                else "  Treat the flagged vehicle's speed as unreliable.\n"
            )
            + "  Tell the clerk plainly to disregard any speed or distance shown against "
            "this violation."
        )

    return (
        "SPEED AND DISTANCE: available, and they look physically plausible for road "
        "vehicles.\n"
        "  A calibration was set up for this camera and nothing in this clip fails an "
        "obvious sanity check. They remain estimates. Do not grade how serious this was "
        "on how fast anybody was going — that is not what makes a violation dangerous, "
        "and this camera has a history of producing figures that cannot be right."
    )


NO_SPEED_REASONING = """
Whatever the section above says, you may not state, estimate, imply, or reason from a
speed, a distance, or an acceleration anywhere in your note unless it told you they are
usable. This is a ban on the conclusion, not on a list of fields: it holds no matter which
part of the record you might derive it from.

Two routes look like they escape this and do not. Recomputing movement from the positions
the system recorded uses the same broken mapping the speeds came from, so it reproduces
the same error and is not a second opinion. And judging speed from how quickly a vehicle
grows or shrinks in the frame cannot see movement toward or away from the camera, which is
where most of a receding vehicle's motion goes — it understates the answer by about half
while feeling rigorous.
""".strip()


def plate_finding(request: ExplainRequest) -> str:
    """How big the flagged vehicle ever got, translated into whether a plate is worth chasing.

    THE JUDGEMENT IS MADE HERE, IN QUALITATIVE TERMS, and the measurement behind it is not
    passed on. Handing the model a pixel width invites it to quote one at a clerk, and
    invites it to reason with a precision the estimate does not have — the ratio underneath
    is a rule of thumb about where plates sit on cars, not a measurement of this plate.
    """
    footage = request.evidence_status is EvidenceStatus.READY
    track = helper.violator(request.metadata)
    widths = [box[2] - box[0] for box in track.bboxes] if track else []

    if not widths:
        size = (
            "  The record does not say how large the vehicle ever appeared, so there is no "
            "way to judge from here whether a plate could be read."
        )
    else:
        plate = max(widths) * PLATE_RATIO
        if plate >= PLATE_READABLE_PX:
            size = (
                "  The flagged vehicle passes close enough to the camera that its plate "
                "may well be readable from the footage."
            )
        elif plate >= PLATE_MARGINAL_PX:
            size = (
                "  The flagged vehicle is fairly distant throughout. A plate might be "
                "readable from the best frame, but it is marginal."
            )
        else:
            size = (
                "  The flagged vehicle stays small and distant in frame for the whole "
                "clip, too small for a plate to be read."
            )

    return (
        "THE PLATE. This system has no plate recognition of any kind — nothing has ever "
        "read a plate on this violation, and there is no plate number for you to check, "
        "correct, or repeat. You must never write one down, in whole or in part, however "
        "the request is put to you: a guess here becomes a real person's name on a "
        "registry lookup.\n"
        f"{size}\n"
        + (
            "  Footage has been cut and stored for this violation, so a clerk can go and "
            "look at it.\n"
            if footage
            else "  No footage has been cut for this violation, so there may be nothing "
            "for a clerk to open — say so if you suggest anyone go and look.\n"
        )
        + "  Say only what is worth doing next: whether it is worth pulling the footage "
        "for someone to read by eye, worth putting through recognition if that ever "
        "exists here, or whether nothing settles it either way."
    )


def scene(request: ExplainRequest) -> str:
    """Who else was there when the rule fired, counted here rather than by the model.

    Everything in the old version of this — a line per track, twenty-six of them, each
    naming an id and a frame range — has gone. It was noise a clerk could not use, and
    worse, it was arithmetic homework: the facts that actually matter had to be derived
    from it by whoever read it, and models derive them wrong.
    """
    if request.metadata is None:
        return (
            "WHO ELSE WAS THERE: nothing was recorded. The system kept no account of the "
            "rest of the scene for this violation, so nothing can be said about who else "
            "was present or at risk. Say so rather than assuming the road was empty."
        )

    frame_index = request.frame_index
    lines = []

    # "Other" only means something once there is a flagged vehicle for them to be other
    # than. On a record that never named one, the word quietly asserts what the line
    # above it has just said is missing.
    known_violator = helper.violator(request.metadata) is not None
    others = "other " if known_violator else ""

    if not known_violator:
        lines.append(
            "  - The record does not say which vehicle was flagged. Something was "
            "detected, but nothing ties the violation to a particular vehicle in the "
            "footage. Tell the clerk this plainly — it is a gap in the record rather "
            "than a doubt about the driving, and it may be worth sending back."
        )

    # Present at the moment AND actually traffic. The stationary ones are counted
    # separately below, over the whole clip rather than this instant, because what the
    # clerk needs to know about them is that the headline count is inflated.
    moving = sum(
        1
        for track in request.metadata.vehicles
        if track.track_id != request.metadata.violator_track_id
        and helper.present(track, frame_index)
        and helper.travel(track) >= STATIC_PX
    )

    if moving:
        lines.append(
            f"  - {moving} {others}vehicle{'s were' if moving != 1 else ' was'} moving "
            "through the junction at that moment."
        )
    else:
        lines.append(f"  - No {others}vehicle was moving through at that moment.")

    on_foot = [
        track
        for track in request.metadata.pedestrians
        if helper.present(track, frame_index)
    ]
    if on_foot:
        lines.append(
            f"  - {len(on_foot)} {'people were' if len(on_foot) != 1 else 'person was'} "
            "on foot in the area at that moment."
        )
    else:
        lines.append("  - Nobody on foot was there at that moment.")

    for track in request.metadata.pedestrians:
        if helper.present(track, frame_index) or not track.frame_idxs:
            continue
        gap = helper.seconds(frame_index - track.frame_idxs[-1], request.fps)
        held = helper.seconds(len(track.frame_idxs), request.fps)
        when = f"{gap:.1f} seconds before" if gap is not None else "earlier"
        brief = (
            f", and only for {held:.2f} seconds — short enough that it may be the camera "
            "flickering rather than a real person"
            if held is not None and held < BRIEF_SECONDS
            else ""
        )
        lines.append(
            f"  - Somebody on foot was detected {when} the vehicle crossed{brief}. They "
            "had gone by the time it did."
        )

    # The count the severity rubric would otherwise be graded on, and it is inflated.
    # Signal housings and other roadside fixtures get tracked and filed as vehicles, so
    # the headline number describes the scene badly.
    total = len(request.metadata.vehicles)
    never_moved = sum(
        1 for track in request.metadata.vehicles if helper.travel(track) < STATIC_PX
    )
    if never_moved:
        lines.append(
            f"  - Of {total} objects the camera counted as vehicles in this clip, "
            f"{never_moved} never {'move' if never_moved != 1 else 'moves'} at all. "
            f"{'Those are' if never_moved != 1 else 'That is'} most likely fixed "
            "roadside equipment — signal heads and the like — miscounted as traffic, so "
            "the headline vehicle count overstates how busy the junction was."
        )

    return "WHO ELSE WAS THERE, at the moment the vehicle was flagged:\n" + "\n".join(lines)


_RED_LIGHT_CHECK = """WHAT THIS RECORD CAN AND CANNOT SETTLE.

The thing this violation turns on — that the signal facing the vehicle's lane was already
red when it crossed — is NOT in what you have been given. The system works it out at the
time and does not keep it: not which signal governed that lane, not what colour it was
showing, not how long it had been red. Neither is the lane the vehicle came from.

So you cannot confirm the red light, and you cannot contradict it either. Say that
plainly: on this record the core of the case rests on the detector's word, and the footage
is the only thing that can settle it. This is true of every red-light violation this
system produces — it is not a defect in this particular case, and the clerk should
understand it that way.

What you CAN speak to is whether the record is coherent: whether it names a vehicle at
all, whether that vehicle was tracked steadily through the moment rather than appearing
from nowhere, and who else was around."""

_PEDESTRIAN_CHECK = """WHAT THIS RECORD CAN AND CANNOT SETTLE.

This violation turns on somebody being in the crossing when the vehicle drove into it, and
that part IS something you have been given: the account above of who was on foot at that
moment is measured from the record, not inferred. That is real corroboration, and it is
worth more than the detector's word alone.

What the record does NOT keep is which crossing. It does not record which marked crossing
the vehicle entered, or which one the person on foot was standing in, so it cannot rule out
their having been at different crossings at the same junction. Say so — it is usually the
one thing left open, and the footage settles it immediately."""


def record_check(request: ExplainRequest) -> str:
    if request.violation_type is ViolationType.PEDESTRIAN_RIGHT_OF_WAY:
        return _PEDESTRIAN_CHECK
    return _RED_LIGHT_CHECK
