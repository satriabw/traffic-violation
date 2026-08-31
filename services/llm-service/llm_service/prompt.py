"""Turning one violation into a note the clerk who reviews it can act on.

WHO READS THIS. A records clerk, with the footage open, deciding whether to approve the
violation, reject it, hold it for a supervisor, or send it back for reprocessing. Not an
engineer, and not the system itself. Every choice below follows from that.

THE MODEL IS NEVER SHOWN AN IDENTIFIER. No track ids, no frame indices, no pixel
measurements reach the prompt — which is why none of them can reach the clerk. Forbidding
them in the output is the weaker version of this and was tried first; a vocabulary the
model never receives is one it cannot leak, and the identifiers carry nothing a clerk
needs. The row keeps `id`, `source_id` and `frame_index` for whoever audits the decision
later, so nothing is actually lost.

THE MODEL IS NEVER ASKED TO COUNT. Everything derived from the track record — who was
present when the rule fired, which speeds are impossible, what never moved — is computed
here and handed over as a finished statement. This is not a stylistic preference. A
three-arm study over this exact data (`research/violation-prompt-study/round3/`) asked two
independent arms to screen twenty-six tracks for implausible speeds; one answered six, the
other seven, and the true figure is nine. One of them had written down that precise failure
mode as a risk before committing it unnoticed. A miscount here lands in a document
supporting enforcement against a real person, and nothing downstream can catch it.

WHAT THIS PROMPT CANNOT ASK FOR. There are no images. The model gets the detector's
structured record and nothing else, so it cannot report what a signal displayed, who was
in the crosswalk, or what the weather was — and a prompt that asks for "observations from
the evidence" without saying so invites exactly those inventions.

Three findings from the earlier rounds remain load-bearing and are preserved:

  * Every severity that came back too high came from a run that believed the speed
    telemetry, and a model handed those numbers will grade on them.

  * Telling a model to "fall back to the trajectory" makes it worse. The trajectory comes
    out of the same broken pixel-to-world mapping as the speeds, one integration apart, so
    recomputing from it reproduces the same answer while feeling like corroboration.

  * Naming the forbidden *inputs* does not work — there is always another field carrying
    the same information. The prohibition has to be on the conclusion.
"""

from shared.models.detection import ViolationType
from shared.models.explanation import ExplainRequest
from shared.models.violation import EvidenceStatus, TrackSummary, ViolationMetadata

SYSTEM = (
    "You are an experienced traffic enforcement officer writing a short note to the "
    "records clerk who will decide what happens to one flagged violation. They will "
    "approve it, reject it, hold it for a supervisor, or send it back for reprocessing "
    "— that decision is theirs and you must not make it for them, or use those words to "
    "steer it. Your job is to tell them what the file actually shows, how much of it you "
    "would stand behind, and what is missing.\n\n"
    "Write for a colleague, not for a computer. Never mention a track number, a frame "
    "number, a pixel measurement, or the name of a field in a database — a clerk does "
    "not know what those are, and you have been given the case in plain terms precisely "
    "so you do not need them. Say 'the flagged vehicle', 'another vehicle', 'someone on "
    "foot', and give times in seconds.\n\n"
    "Your findings may support enforcement action against a real person, so every claim "
    "must be supported by what you were given. Where it does not settle something, say "
    "so plainly — that is useful to the clerk, not a gap to paper over."
)

# Heuristics, all three tuned on a single scene and all three documented as such. They
# decide what the model is told, so getting one wrong is a wrong statement rather than a
# wrong emphasis.
#
# Movement below this across a track's whole life is something that is not traffic. In the
# one scene measured, the confirmed-stationary objects topped out at under 2px of travel
# and the slowest genuinely moving vehicle managed 35px, so the gap is wide here — but a
# rolling stop or a slow encroachment is a violation where barely moving is the whole
# point, and this would misfile it. Revisit before trusting it on a second site.
_STATIC_PX = 15.0
# No road vehicle sustains this. Generous on purpose: the job is catching a derivation
# that has failed outright, not policing the speed limit, and a tighter bound would start
# making judgements about driving.
_IMPLAUSIBLE_MPS = 45.0
# A detection this short is more likely to be the tracker flickering than a person. Worth
# telling the clerk, because "two pedestrians were present" reads very differently from
# "two detections lasting a twentieth of a second each".
_BRIEF_SECONDS = 0.2
# A rear plate runs about a fifth of the width of the car carrying it. Measured on this
# camera: a 154px-wide vehicle carried a ~30px plate, which no upscaling made readable.
_PLATE_RATIO = 0.2
# Rough widths a plate has to reach before anybody can read it — recognition wants the
# larger, a person squinting at good footage can sometimes manage the smaller.
_PLATE_READABLE_PX = 100.0
_PLATE_MARGINAL_PX = 60.0


def _seconds(frame_index: int | None, fps: float | None) -> float | None:
    """A frame index as a time into the clip, or None if nothing can convert it."""
    if frame_index is None or not fps or fps <= 0:
        return None
    return frame_index / fps


def _moment(request: ExplainRequest) -> str:
    """Where in the footage this happened, in the terms somebody scrubbing it would use.

    Falls back to naming no position at all rather than to the frame index. A source
    nobody probed has no frame rate, and a clerk handed "frame 159" has been given a
    number they cannot act on and would have to ask an engineer to interpret.
    """
    at = _seconds(request.frame_index, request.fps)
    if at is None:
        return "at the moment the vehicle was flagged"
    return f"{at:.1f} seconds into the footage"


def _travel(track: TrackSummary) -> float:
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


def _present(track: TrackSummary, frame_index: int | None) -> bool:
    """Was this object there at the moment the rule fired?

    MEMBERSHIP, NOT SPAN. Asking whether the violation frame falls between a track's first
    and last sample is a different and much weaker question, and on this data it is wrong
    often enough to matter: the evidence window ends on the violation frame, so every
    object still on screen when the buffer closed passes that test — seventeen of
    twenty-six in the scene this was measured on. Tracks also have gaps.
    """
    return frame_index is not None and frame_index in track.frame_idxs


def _implausible(track: TrackSummary) -> float | None:
    """This track's top speed, if it is one nothing on a road could have done."""
    speeds = [speed for speed in track.speed if speed is not None]
    if not speeds or max(speeds) <= _IMPLAUSIBLE_MPS:
        return None
    return max(speeds)


def _violator(metadata: ViolationMetadata | None) -> TrackSummary | None:
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


def _speed_finding(request: ExplainRequest) -> str:
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
    failures = [speed for speed in (_implausible(track) for track in tracks) if speed]

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


_NO_SPEED_REASONING = """
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


def _plate_finding(request: ExplainRequest) -> str:
    """How big the flagged vehicle ever got, translated into whether a plate is worth chasing.

    THE JUDGEMENT IS MADE HERE, IN QUALITATIVE TERMS, and the measurement behind it is not
    passed on. Handing the model a pixel width invites it to quote one at a clerk, and
    invites it to reason with a precision the estimate does not have — the ratio underneath
    is a rule of thumb about where plates sit on cars, not a measurement of this plate.
    """
    footage = request.evidence_status is EvidenceStatus.READY
    track = _violator(request.metadata)
    widths = [box[2] - box[0] for box in track.bboxes] if track else []

    if not widths:
        size = (
            "  The record does not say how large the vehicle ever appeared, so there is no "
            "way to judge from here whether a plate could be read."
        )
    else:
        plate = max(widths) * _PLATE_RATIO
        if plate >= _PLATE_READABLE_PX:
            size = (
                "  The flagged vehicle passes close enough to the camera that its plate "
                "may well be readable from the footage."
            )
        elif plate >= _PLATE_MARGINAL_PX:
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


def _scene(request: ExplainRequest) -> str:
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
    known_violator = _violator(request.metadata) is not None
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
        and _present(track, frame_index)
        and _travel(track) >= _STATIC_PX
    )

    if moving:
        lines.append(
            f"  - {moving} {others}vehicle{'s were' if moving != 1 else ' was'} moving "
            "through the junction at that moment."
        )
    else:
        lines.append(f"  - No {others}vehicle was moving through at that moment.")

    on_foot = [
        track for track in request.metadata.pedestrians if _present(track, frame_index)
    ]
    if on_foot:
        lines.append(
            f"  - {len(on_foot)} {'people were' if len(on_foot) != 1 else 'person was'} "
            "on foot in the area at that moment."
        )
    else:
        lines.append("  - Nobody on foot was there at that moment.")

    for track in request.metadata.pedestrians:
        if _present(track, frame_index) or not track.frame_idxs:
            continue
        gap = _seconds(frame_index - track.frame_idxs[-1], request.fps)
        held = _seconds(len(track.frame_idxs), request.fps)
        when = f"{gap:.1f} seconds before" if gap is not None else "earlier"
        brief = (
            f", and only for {held:.2f} seconds — short enough that it may be the camera "
            "flickering rather than a real person"
            if held is not None and held < _BRIEF_SECONDS
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
        1 for track in request.metadata.vehicles if _travel(track) < _STATIC_PX
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


def _record_check(request: ExplainRequest) -> str:
    if request.violation_type is ViolationType.PEDESTRIAN_RIGHT_OF_WAY:
        return _PEDESTRIAN_CHECK
    return _RED_LIGHT_CHECK


def build_prompt(request: ExplainRequest) -> str:
    return f"""Write the clerk a short note on one flagged violation.

WHAT YOU HAVE: an account of the file, already summarised for you in plain terms. There is
no imagery. You cannot see the signal, the road, the weather, or anyone on foot, and you
must not write as though you can. You also have no case notes, no vehicle registration,
and no history for this driver.

You have deliberately NOT been given track numbers, frame numbers, or pixel measurements.
Everything countable has been counted for you and is stated below. Do not ask for them, do
not estimate them, and do not invent them.

THE VIOLATION
- What was flagged: {request.violation_type.value.replace("_", " ")}
- Where: {request.site_name}
- When: {request.detected_at.isoformat()}
- Where in the footage: {_moment(request)}

{_speed_finding(request)}

{_NO_SPEED_REASONING}

{_scene(request)}

{_record_check(request)}

{_plate_finding(request)}

BEFORE YOU ENDORSE IT, TEST IT. Detectors get things wrong: they track the wrong object,
put a vehicle in the wrong lane, or fire on one that entered lawfully and was still
clearing the junction when the lights changed. Ask whether what you have been told is
consistent with the offence being alleged.

Only say the flag does not stand if something here actively contradicts it. Something the
record merely fails to settle is not a contradiction — it lowers how much of this you can
stand behind, and you say so, but rejecting a violation is the clerk's decision and "we
cannot tell from here" is a different statement from "this did not happen".

HOW MUCH OF THIS YOU WOULD STAND BEHIND. Choose one:
  STRONG - the record establishes the offence on its own, without leaning on the
           detector's word. On the type of violation you are looking at this is currently
           unreachable, because the system does not keep the facts that would do it. Do
           not award it. If you find yourself reaching for it, you have over-claimed.
  MEDIUM - consistent with the flag, with something load-bearing taken on trust.
  WEAK   - the record cannot get past the detector's assertion, or something needed is
           missing or contradictory.

Say what carries that judgement in evidence_basis, in the clerk's terms: what the footage
would have to settle, and what the record already establishes on its own.

HOW SERIOUS IT WAS. This is a separate question from how well evidenced it is — a serious
event can be thinly evidenced, and a trivial one established beyond doubt. Grade on who
was actually put at risk, using the account of the scene above, and never on speed.

HIGH   - somebody on foot was there when the vehicle went through.
MEDIUM - other traffic was moving through at the time.
LOW    - nobody else was there.

Name in severity_basis which of those you found.

OBSERVATIONS. Two to four at most, and a line earns its place only if the clerk could not
have worked it out themselves from the file. A count read back to them is not an
observation. Two facts that mean something once you put them together — objects counted as
traffic that never move, somebody on foot detected for a twentieth of a second, a record
that does not name the vehicle it accuses — are exactly what this is for. If a line would
not change how the clerk handles the case, leave it out.

WHAT NOT TO TRUST goes in evidence_concerns, phrased as what to do about it rather than
how it works. "Disregard any speed shown on this record — the camera's distance
calibration is faulty" is useful. How many objects failed which threshold is how that was
worked out, and is not the clerk's problem. Do not let a doubt quietly change how serious
you said it was, and do not leave one out because it is inconvenient.

Write every part of your answer for the clerk: plain professional English, no system
vocabulary, no field names, no numbers they cannot act on."""
