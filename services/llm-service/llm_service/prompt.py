"""Turning one violation into the text a model is asked to explain.

The wording here is not arbitrary. It comes out of a ten-arm study over one real
violation, and three findings from it are load-bearing:

  * Every severity that came back too high came from a run that believed the speed
    telemetry. On a violation with no calibration pinned those numbers are ~9x wrong,
    and a model handed them will grade on them.

  * Telling a model to "fall back to the trajectory" makes it worse, not better. The
    trajectory comes out of the same broken pixel-to-world mapping as the speeds, one
    integration apart, so recomputing from it reproduces the same wrong answer while
    feeling like corroboration.

  * Naming the forbidden *inputs* does not work — there is always another field
    carrying the same information, and bounding boxes are the one everybody forgets.
    The prohibition has to be on the conclusion.

WHAT THIS PROMPT CANNOT ASK FOR. There are no images. The model gets the detector's
structured record and nothing else, so it cannot report what a signal displayed, who
was in the crosswalk, or what the weather was — and a prompt that asks for
"observations from the evidence" without saying so invites exactly those inventions.
Everything below is scoped to what a track record and a geometry document can support.
"""

import json

from shared.models.explanation import ExplainRequest

SYSTEM = (
    "You are a traffic enforcement assistant. Your findings may support enforcement "
    "action against a real person, so every claim you make must be supported by the "
    "evidence you were given. Where the evidence does not settle something, say so "
    "rather than filling the gap."
)

# Withheld together, because they are the same number twice. Splitting them — offering
# the trajectory as a fallback when the speeds are untrusted — is the failure the study
# found hardest to see from inside: the arm that did it recomputed four different
# windows, got the same impossible answer every time, and read the agreement as
# confirmation.
_NO_MOTION = """- Vehicle trajectories and speeds: WITHHELD.
  This violation was recorded with no camera calibration pinned, so no valid
  pixel-to-world mapping exists for this footage.

  You may not state, estimate, imply, or reason from a speed, a distance, or an
  acceleration anywhere in your answer. This is a ban on the conclusion, not on a list
  of fields: it holds no matter which part of the record you might derive it from,
  the bounding boxes included.

  Two routes look like they escape this and do not. The trajectory is produced by the
  same mapping as the speeds, so recomputing displacement from it reproduces the same
  error and is not a second opinion. And converting bounding-box movement into metres
  using the vehicle's apparent size corrects the scale but not the projection — it
  cannot see motion along the camera's axis, which is where most of a receding
  vehicle's motion goes, and it understates the answer by about half while feeling
  rigorous.

  Do not treat speed as a severity factor."""


def _motion(request: ExplainRequest) -> str:
    if request.calibration_id is None:
        return _NO_MOTION
    tracks = request.metadata.vehicles if request.metadata else []
    return (
        "- Vehicle trajectories and speeds: in the track record below, derived through "
        f"the site's active camera calibration ({request.calibration_id}) and therefore "
        "usable.\n"
        f"- Tracks carrying motion data: {len(tracks)}.\n"
        "  They remain estimates. If a value is physically implausible for a road "
        "vehicle, say so in evidence_concerns rather than reasoning from it."
    )


def _tracks(request: ExplainRequest) -> str:
    """The scene as the detector recorded it, summarised rather than dumped.

    Every sample of every track is ~13.5KB and says little a reader of the explanation
    needs. What decides severity is who else was there and whether they were moving,
    which is a count and a span per track.
    """
    if request.metadata is None:
        return "- Track record: MISSING. Nothing was recorded about who else was present."

    lines = [
        f"- The detector convicted track {request.metadata.violator_track_id}."
        if request.metadata.violator_track_id is not None
        else "- The detector did not record which track it convicted.",
        f"- Vehicles on the scene: {len(request.metadata.vehicles)}.",
        f"- Pedestrians on the scene: {len(request.metadata.pedestrians)}.",
    ]
    for label, tracks in (
        ("vehicle", request.metadata.vehicles),
        ("pedestrian", request.metadata.pedestrians),
    ):
        for track in tracks:
            frames = track.frame_idxs
            span = f"frames {frames[0]}-{frames[-1]}" if frames else "no frames"
            lines.append(
                f"  - {label} track {track.track_id}: {span}, {len(frames)} samples"
            )
    return "\n".join(lines)


def build_prompt(request: ExplainRequest) -> str:
    configuration = (
        json.dumps(request.configuration, indent=2, sort_keys=True)
        if request.configuration
        else "MISSING — the regions and rules this was judged against are not available."
    )
    return f"""Explain one detected traffic violation.

WHAT YOU HAVE: the detector's structured record. There is no imagery. You cannot see
the signal, the road, the weather, or anyone on foot, and you must not write as though
you can. Do not describe what is visible; describe what the record establishes, and
name what it leaves open.

Violation details:
- Type: {request.violation_type.value}
- Detected at: {request.detected_at.isoformat()}
- Site: {request.site_name}
- Frame index of the violation: {request.frame_index}
{_motion(request)}

The scene the detector recorded:
{_tracks(request)}

The site configuration in force when this was judged — lane polygons, traffic-light
regions, and the rules that were enabled:
{configuration}

Before endorsing the detection, test it. Detectors produce false positives: they track
the wrong object, mis-assign a lane, or fire on a vehicle that entered legally and was
still clearing the junction. Ask whether the convicted track's presence in the record
is consistent with the rule that fired and the regions it was judged against. If the
record does not sustain the flag, set flag_sustained to false and say why.

Severity. Choose the HIGHEST band whose conditions the record actually establishes.
Do not grade on anything you had to assume, and do not grade on speed.

HIGH   - Pedestrians were on the scene during the violation; or the record shows
         other tracks in the conflict area through the moment the rule fired.
MEDIUM - Other road users were present and tracked, but the record does not establish
         that any of them was in conflict.
LOW    - The record shows no other road user present: the violation affected nobody
         who can be identified in it.

State in severity_basis which of those conditions you found, in the record, by name.

Where the record leaves something open — a missing track blob, an absent configuration,
a value that cannot be right — put it in evidence_concerns. Do not let it silently
change the severity, and do not omit it because it is inconvenient."""
