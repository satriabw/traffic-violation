"""The numbers the prompt's statements turn on.

ALL OF THESE WERE TUNED ON A SINGLE SCENE, and every one of them decides what the model
is told rather than how it says it — so getting one wrong produces a wrong statement to a
clerk, not a wrong emphasis. They are gathered here so that the evidence behind each is
in one place to argue with, and so a second site can be fitted without reading the prose
that consumes them.

Nothing here is a tuning knob for output quality. Read the comment before moving a value.
"""

# Movement below this across a track's whole life is something that is not traffic. In the
# one scene measured, the confirmed-stationary objects topped out at under 2px of travel
# and the slowest genuinely moving vehicle managed 35px, so the gap is wide here — but a
# rolling stop or a slow encroachment is a violation where barely moving is the whole
# point, and this would misfile it. Revisit before trusting it on a second site.
STATIC_PX = 15.0

# No road vehicle sustains this. Generous on purpose: the job is catching a derivation
# that has failed outright, not policing the speed limit, and a tighter bound would start
# making judgements about driving.
IMPLAUSIBLE_MPS = 45.0

# A detection this short is more likely to be the tracker flickering than a person. Worth
# telling the clerk, because "two pedestrians were present" reads very differently from
# "two detections lasting a twentieth of a second each".
BRIEF_SECONDS = 0.2

# A rear plate runs about a fifth of the width of the car carrying it. Measured on this
# camera: a 154px-wide vehicle carried a ~30px plate, which no upscaling made readable.
PLATE_RATIO = 0.2

# Rough widths a plate has to reach before anybody can read it — recognition wants the
# larger, a person squinting at good footage can sometimes manage the smaller.
PLATE_READABLE_PX = 100.0
PLATE_MARGINAL_PX = 60.0
