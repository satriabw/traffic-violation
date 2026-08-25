import numpy as np

from violation_detector.traffic_light import (
    GREEN,
    RED,
    UNKNOWN,
    YELLOW,
    state_from_pixels,
)

# BGR, the order a frame arrives in. Bright enough to be unmistakably lit.
LIT = {RED: (0, 0, 255), YELLOW: (0, 255, 255), GREEN: (0, 255, 0)}
# A lamp that is off is not black — it is a dark lens with the housing behind it.
OFF = (18, 18, 18)


def light(lit: str | None, vertical: bool = True, size: int = 30) -> np.ndarray:
    """A three-lamp housing with one lamp lit, or none."""
    lamps = [LIT[lit] if lit == name else OFF for name in (RED, YELLOW, GREEN)]
    long, short = size, size // 3
    patch = np.zeros((long, short, 3) if vertical else (short, long, 3), dtype=np.uint8)
    for i, colour in enumerate(lamps):
        span = slice(i * long // 3, (i + 1) * long // 3)
        if vertical:
            patch[span, :] = colour
        else:
            patch[:, span] = colour
    return patch


def test_a_vertical_light_is_read_from_the_top_down():
    assert state_from_pixels(light(RED)) == RED
    assert state_from_pixels(light(YELLOW)) == YELLOW
    assert state_from_pixels(light(GREEN)) == GREEN


def test_a_horizontal_light_is_read_from_the_left():
    assert state_from_pixels(light(RED, vertical=False)) == RED
    assert state_from_pixels(light(YELLOW, vertical=False)) == YELLOW
    assert state_from_pixels(light(GREEN, vertical=False)) == GREEN


def test_a_light_with_nothing_lit_is_unknown_rather_than_a_colour():
    # Every lamp dark. Guessing here would put a red light on a junction that has
    # none, and a rule downstream cannot tell a guess from a reading.
    assert state_from_pixels(light(None)) == UNKNOWN


def test_a_lamp_only_just_bright_enough_still_reads():
    # The threshold is a mean over the lamp's third, not a peak, so a dim but lit
    # lamp at dusk is not thrown away.
    patch = light(None)
    patch[: patch.shape[0] // 3, :] = (40, 40, 60)
    assert state_from_pixels(patch) == RED


def test_a_light_annotated_off_the_edge_of_the_frame_is_unknown():
    # `crop` clips to the frame, so a region annotated past the border arrives here
    # with no pixels at all. That is a frame this light cannot be judged on, not a
    # crash mid-video.
    assert state_from_pixels(np.zeros((0, 12, 3), dtype=np.uint8)) == UNKNOWN
    assert state_from_pixels(np.zeros((12, 0, 3), dtype=np.uint8)) == UNKNOWN


def test_a_light_too_small_to_divide_is_unknown():
    # Two pixels tall cannot be split into three lamps. The source pipeline sliced it
    # anyway, took the mean of an empty array, and compared NaNs.
    assert state_from_pixels(np.full((2, 1, 3), 255, dtype=np.uint8)) == UNKNOWN


def test_a_frame_that_is_not_colour_is_unknown():
    assert state_from_pixels(np.full((30, 10), 255, dtype=np.uint8)) == UNKNOWN


def test_a_square_housing_is_read_as_horizontal():
    patch = np.full((9, 9, 3), OFF, dtype=np.uint8)
    patch[:, :3] = LIT[RED]
    assert state_from_pixels(patch) == RED


def test_the_brightest_lamp_wins_when_two_are_lit():
    # An amber-to-red transition caught mid-change, or a reflection on a neighbouring
    # lens. Position and brightness are all this has to go on.
    patch = light(None)
    third = patch.shape[0] // 3
    patch[:third, :] = (0, 0, 120)
    patch[third : 2 * third, :] = (0, 255, 255)
    assert state_from_pixels(patch) == YELLOW
