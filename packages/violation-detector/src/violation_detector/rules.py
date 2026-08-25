"""Whether what is known amounts to a violation.

The third stage, and the only one that is a statement about traffic law rather than
about pixels or bookkeeping. Plain functions over frozen values: no frame, no OpenCV,
no tracker, nothing to set up. A truth table can be written straight against them, and
that is the point — this is the part where being wrong means accusing somebody.

The source pipeline built these as a list of closures, one per light, assembled by a
`add_rule` loop and evaluated with `any(...)`. The machinery was there to close over
each light's lane list; iterating the same mapping directly says it in one place, and
lets the conditions that do not depend on the light be checked once instead of once
per light.
"""

from violation_detector.context import LightState, Occupancy, VehicleState
from violation_detector.traffic_light import RED


def is_running_red(
    vehicle: VehicleState,
    lights: dict[str, LightState],
    controlled_lanes: dict[str, tuple[str, ...]],
) -> bool:
    """Did this vehicle just cross into the box against a red light?

    Five conditions, all of which must hold for the same light:

      1. the vehicle is inside the box past the stop line;
      2. it was not inside it on an earlier frame — so this is the crossing itself,
         reported once, and not every frame it spends in the box;
      3. the light governing the lane it came from is red *now*;
      4. that lane is one this light governs — a vehicle turning out of a lane some
         other light controls is that light's business;
      5. it entered the box no earlier than the moment that light turned red. This is
         what separates running a red from being caught inside the junction when the
         light changed, which is not the same offence and is often not one at all.
    """
    # Cheap and light-independent, so they settle it before any light is consulted.
    if not vehicle.in_roi or vehicle.prev_in_roi:
        return False

    for light_id, lanes in controlled_lanes.items():
        light = lights.get(light_id)
        # A light the scene declares but this frame has nothing to say about. It
        # cannot be shown to be red, so it cannot convict.
        if light is None or light.state != RED:
            continue
        if vehicle.last_lane not in lanes:
            continue
        if vehicle.enter_roi_frame >= light.start_red:
            return True

    return False


def failing_to_yield(occupancy: Occupancy) -> tuple[int, ...]:
    """Which vehicles drove into this crossing while somebody was in it.

    Two conditions, and the whole rule:

      1. somebody vulnerable is in this crossing — a pedestrian, a cyclist, anyone the
         traffic is supposed to give way to;
      2. a vehicle entered it on this frame, and was not already inside on the last
         one.

    The second is what makes this about *yielding* rather than about proximity. A car
    already stopped in the box when a pedestrian steps off the kerb has not failed to
    give way to them; a car that drives in while they are crossing has. Without it the
    rule would also report the same car on every frame the two are in the box together,
    and the count would grow with how long the pedestrian took.

    Every entering vehicle is reported, not just one. If three cars push into an
    occupied crossing on the same frame, three drivers failed to yield.

    Co-presence in ONE crossing: this is asked per region of interest, so a pedestrian
    at one crossing cannot make violators of vehicles entering another.
    """
    if not occupancy.pedestrians:
        return ()
    return tuple(
        vehicle.track_id
        for vehicle in occupancy.vehicles
        if vehicle.in_roi and not vehicle.prev_in_roi
    )
