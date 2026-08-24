"""What a light is showing, read from the pixels inside its polygon.

The light is not detected — it is annotated, once, in the site's configuration, and
this reads the colour out of that fixed rectangle on every frame. A junction's lights
do not move, and a detector that has to find them again 30 times a second is a
detector that can lose one.

HOW IT DECIDES. Split the lamp housing into three along its long axis, take the mean
brightness of each third, and name the brightest one by its position: red, amber,
green from the top of a vertical light or the left of a horizontal one. If even the
brightest third is dim, nothing is lit and the answer is UNKNOWN.

WHAT THAT BUYS, AND WHAT IT COSTS. It needs no training, no model and no per-site
tuning, and it is right about the overwhelmingly common case — one lamp lit, the other
two dark. What it cannot do is notice that the bright third is the *wrong colour*: sun
on the top lens of a dark light reads as RED, and a light seen so far away that its
lamps blur together reads as whatever the housing reflects. Position is doing all the
work. A rule downstream is therefore never more certain than this function is, which
is why a light state is checked alongside a vehicle's lane and its entry frame rather
than on its own.

The source pipeline also carried a `BY_COLOR_HISTOGRAM` method that raised
NotImplementedError on every call, and a `method=` parameter to choose between it and
this one. Both are gone; one method that works needs no selector.
"""

import numpy as np

# Lowercase throughout the package — these travel next to "red_light_running" and
# "bottom_center", not next to an enum in someone's database.
RED = "red"
YELLOW = "yellow"
GREEN = "green"
UNKNOWN = "unknown"

# Top to bottom, or left to right. The order lamps are installed in.
#
# NOT ALWAYS LEFT TO RIGHT. A horizontal light mounted for traffic coming the other
# way runs green-amber-red, and nothing in the picture says which one this is. The
# common case is assumed; the day a site needs the other, it is a field on the light
# in the configuration document, not a guess made here.
LAMPS = (RED, YELLOW, GREEN)

# Mean brightness, 0-255, below which a third counts as unlit. Every lamp being under
# it means the light is off, obscured, or too small to read — all of which are UNKNOWN
# rather than a colour, because a rule that fires on a misread light is worse than one
# that does not fire.
BRIGHTNESS_THRESHOLD = 50.0

# A lamp needs at least one row of pixels to have a mean. Below this the crop is too
# small to divide, which is a light too far away to read.
MINIMUM_EXTENT = 3


def state_from_pixels(patch: np.ndarray) -> str:
    """The colour a light is showing, given the pixels of its housing.

    Never raises. A light annotated off the edge of the frame, or one small enough to
    have collapsed to nothing, returns UNKNOWN — the same answer as a light that is
    simply dark. Every one of those is a frame this light cannot be judged on, and a
    video is thousands of frames long.
    """
    if patch.ndim != 3 or patch.shape[2] != 3 or patch.size == 0:
        return UNKNOWN

    # Brightness is the HSV value channel, which for 8-bit BGR is exactly the largest
    # of the three channels — that is how OpenCV computes it. Taking the max directly
    # gives an identical answer without a colour conversion per light per frame, and
    # without caring which order the channels arrived in.
    brightness = patch.max(axis=2)

    height, width = brightness.shape
    # Taller than wide is a vertical light. A square one is treated as horizontal,
    # which is arbitrary, but a square housing is a light too small to read anyway and
    # will almost always be UNKNOWN on the threshold below.
    vertical = height > width
    extent = height if vertical else width
    if extent < MINIMUM_EXTENT:
        return UNKNOWN

    third, two_thirds = extent // 3, 2 * extent // 3
    if vertical:
        lamps = (brightness[:third], brightness[third:two_thirds], brightness[two_thirds:])
    else:
        lamps = (brightness[:, :third], brightness[:, third:two_thirds], brightness[:, two_thirds:])

    means = [float(lamp.mean()) for lamp in lamps]
    if max(means) < BRIGHTNESS_THRESHOLD:
        return UNKNOWN
    return LAMPS[int(np.argmax(means))]
