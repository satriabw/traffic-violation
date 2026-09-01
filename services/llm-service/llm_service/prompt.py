"""Turning one violation into a note the clerk who reviews it can act on.

WHO READS THIS. A records clerk, with the footage open, deciding whether to approve the
violation, reject it, hold it for a supervisor, or send it back for reprocessing. Not an
engineer, and not the system itself. Every choice below follows from that.

WHERE THE REST OF IT IS. This module holds the voice and the assembled document; the three
beside it hold what the document is built from, because they change for different reasons
and on different evidence:

  * `llm_service.constant` — the tuned numbers, each with the measurement behind it.
  * `llm_service.helper` — the arithmetic over the track record, done here so the model
    never does it. See that module for why: a study over this data had two arms miscount
    the same twenty-six tracks in two different directions.
  * `llm_service.template` — the wording of each section, which is what a round of the
    study usually sends back for revision.

THE MODEL IS NEVER SHOWN AN IDENTIFIER. No track ids, no frame indices, no pixel
measurements reach the prompt — which is why none of them can reach the clerk. Forbidding
them in the output is the weaker version of this and was tried first; a vocabulary the
model never receives is one it cannot leak, and the identifiers carry nothing a clerk
needs. The row keeps `id`, `source_id` and `frame_index` for whoever audits the decision
later, so nothing is actually lost.

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

from shared.models.explanation import ExplainRequest

from llm_service import helper, template

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
- Where in the footage: {helper.moment(request.frame_index, request.fps)}

{template.speed_finding(request)}

{template.NO_SPEED_REASONING}

{template.scene(request)}

{template.record_check(request)}

{template.plate_finding(request)}

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
