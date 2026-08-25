import pytest

from violation_detector.context import NEVER, LightState, VehicleState
from violation_detector.rules import is_running_red
from violation_detector.traffic_light import GREEN, RED, UNKNOWN, YELLOW

# One light governing one lane, which is the shape every case below varies from.
CONTROLS = {"tl_1": ("lane_1",)}


def car(
    *,
    lane: str | None = "lane_1",
    in_roi: bool = True,
    prev_in_roi: bool = False,
    enter_roi_frame: int = 20,
) -> VehicleState:
    """A vehicle mid-violation. Each test spoils exactly one condition."""
    return VehicleState(
        track_id=7,
        last_lane=lane,
        in_roi=in_roi,
        prev_in_roi=prev_in_roi,
        enter_roi_frame=enter_roi_frame,
    )


def lights(state: str = RED, start_red: int = 10) -> dict[str, LightState]:
    return {"tl_1": LightState(state=state, start_red=start_red)}


def test_a_vehicle_crossing_on_red_from_a_controlled_lane_is_reported():
    assert is_running_red(car(), lights(), CONTROLS) is True


def test_a_vehicle_still_short_of_the_line_is_not_reported():
    assert is_running_red(car(in_roi=False), lights(), CONTROLS) is False


def test_a_crossing_is_reported_once_and_not_on_every_frame_after():
    # The condition the source pipeline could not evaluate at all: its context builder
    # never put `prev_in_roi` in the dict its rule read, so the first vehicle to reach
    # a controlled lane raised KeyError.
    assert is_running_red(car(prev_in_roi=True), lights(), CONTROLS) is False


def test_a_vehicle_from_a_lane_this_light_does_not_govern_is_not_reported():
    # Turning out of a lane some other light controls. Not this light's business.
    assert is_running_red(car(lane="lane_9"), lights(), CONTROLS) is False


def test_a_vehicle_never_seen_in_any_lane_is_not_reported():
    # Nothing ties it to a light, so nothing can convict it.
    assert is_running_red(car(lane=None), lights(), CONTROLS) is False


@pytest.mark.parametrize("state", [GREEN, YELLOW, UNKNOWN])
def test_a_light_that_is_not_red_reports_nothing(state):
    assert is_running_red(car(), lights(state=state, start_red=NEVER), CONTROLS) is False


def test_a_vehicle_caught_inside_when_the_light_changed_is_not_reported():
    # It entered on frame 5; the light went red on frame 10. Being stuck in the
    # junction when the light changes is a different thing from running it, and
    # usually not an offence at all.
    assert is_running_red(car(enter_roi_frame=5), lights(start_red=10), CONTROLS) is False


def test_a_vehicle_entering_on_the_very_frame_the_light_turned_is_reported():
    # The boundary. Red is red on the frame it becomes red.
    assert is_running_red(car(enter_roi_frame=10), lights(start_red=10), CONTROLS) is True


def test_a_light_the_frame_said_nothing_about_cannot_convict():
    # Defensive: every configured light is observed on every frame, so this should be
    # unreachable. If it ever is reachable, silence is the right answer.
    assert is_running_red(car(), {}, CONTROLS) is False


def test_a_light_governing_several_lanes_covers_all_of_them():
    controls = {"tl_1": ("lane_1", "lane_2", "lane_3")}
    assert is_running_red(car(lane="lane_3"), lights(), controls) is True


def test_only_the_light_governing_the_vehicles_lane_is_consulted():
    # Two lights, and the red one governs a lane this vehicle was never in. A rule
    # that asked "is any light red?" would report it.
    controls = {"tl_1": ("lane_1",), "tl_2": ("lane_2",)}
    states = {
        "tl_1": LightState(state=GREEN, start_red=NEVER),
        "tl_2": LightState(state=RED, start_red=10),
    }
    assert is_running_red(car(lane="lane_1"), states, controls) is False
    assert is_running_red(car(lane="lane_2"), states, controls) is True
