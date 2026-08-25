from violation_detector.context import (
    NEVER,
    PedestrianRightOfWayContext,
    RedLightRunningContext,
)
from violation_detector.observations import (
    CrossingObservation,
    LightObservation,
    Observation,
    Occupant,
    VehicleObservation,
)
from violation_detector.traffic_light import GREEN, RED, UNKNOWN, YELLOW


def seen(
    frame_index: int,
    *,
    light: str | None = None,
    lane: str | None = None,
    roi: str | None = None,
    track_id: int = 7,
) -> tuple[Observation, int]:
    """One frame: one light in the given state, one vehicle in the given places."""
    return (
        Observation(
            lights=(LightObservation(id="tl_1", state=light),) if light else (),
            vehicles=(VehicleObservation(track_id=track_id, lane=lane, roi=roi),),
        ),
        frame_index,
    )


def vehicle(context, track_id: int = 7):
    return next(v for v in context.vehicles if v.track_id == track_id)


def test_a_light_that_turns_red_records_the_frame_it_changed_on():
    builder = RedLightRunningContext()

    builder.update(*seen(10, light=GREEN))
    builder.update(*seen(11, light=YELLOW))
    context = builder.update(*seen(12, light=RED))

    assert context.lights["tl_1"].state == RED
    assert context.lights["tl_1"].start_red == 12


def test_a_light_that_stays_red_keeps_the_frame_it_first_turned():
    # The whole point of the field: a vehicle entering on frame 40 has to be compared
    # against when the light changed, not against now.
    builder = RedLightRunningContext()

    builder.update(*seen(12, light=RED))
    for frame_index in range(13, 40):
        context = builder.update(*seen(frame_index, light=RED))

    assert context.lights["tl_1"].start_red == 12


def test_a_light_that_cycles_records_the_new_red():
    builder = RedLightRunningContext()

    builder.update(*seen(12, light=RED))
    builder.update(*seen(20, light=GREEN))
    context = builder.update(*seen(30, light=RED))

    assert context.lights["tl_1"].start_red == 30


def test_a_light_that_has_not_been_red_has_no_red_frame():
    builder = RedLightRunningContext()

    context = builder.update(*seen(1, light=GREEN))

    assert context.lights["tl_1"].start_red == NEVER


def test_a_light_first_seen_already_red_dates_from_that_frame():
    # The earliest defensible answer. It makes every vehicle already inside the box
    # look like it entered on green, which is the conservative direction.
    builder = RedLightRunningContext()

    context = builder.update(*seen(500, light=RED))

    assert context.lights["tl_1"].start_red == 500


def test_a_light_that_cannot_be_read_is_still_reported():
    # A rule must be able to tell "not red" from "we could not see it".
    builder = RedLightRunningContext()

    context = builder.update(*seen(1, light=UNKNOWN))

    assert context.lights["tl_1"].state == UNKNOWN


def test_a_vehicle_keeps_the_last_lane_it_was_seen_in():
    # The reason this stage exists. Once a vehicle is past the stop line it is in no
    # lane at all, and the lane that decides which light governs it is gone from the
    # frame — but not from here.
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1"))
    context = builder.update(*seen(2, lane=None, roi="roi_1"))

    assert vehicle(context).last_lane == "lane_1"
    assert vehicle(context).in_roi is True


def test_a_vehicle_changing_lanes_is_remembered_by_the_newer_one():
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1"))
    context = builder.update(*seen(2, lane="lane_2"))

    assert vehicle(context).last_lane == "lane_2"


def test_entering_the_box_records_the_frame_and_is_not_yet_a_repeat():
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1"))
    context = builder.update(*seen(2, lane="lane_1", roi="roi_1"))

    assert context.vehicles[0].enter_roi_frame == 2
    # False on the entry frame — this is the crossing, and the rule may fire.
    assert context.vehicles[0].prev_in_roi is False


def test_staying_in_the_box_is_a_repeat_from_the_next_frame_on():
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1"))
    builder.update(*seen(2, lane="lane_1", roi="roi_1"))
    context = builder.update(*seen(3, roi="roi_1"))

    assert context.vehicles[0].prev_in_roi is True
    assert context.vehicles[0].enter_roi_frame == 2


def test_a_vehicle_first_seen_already_inside_the_box_can_never_be_reported():
    # Deliberate, and carried over from the source pipeline: we never saw it enter, so
    # we cannot say it entered on red. It is also why the job's chunks overlap — a
    # vehicle mid-crossing at a boundary would otherwise be invisible to both.
    builder = RedLightRunningContext()

    context = builder.update(*seen(1, lane="lane_1", roi="roi_1"))

    assert context.vehicles[0].prev_in_roi is True
    assert context.vehicles[0].enter_roi_frame == 1


def test_leaving_the_box_does_not_clear_the_entry():
    # A known limit rather than a happy accident: one track is reported for its first
    # crossing only, however many times it goes round.
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1"))
    builder.update(*seen(2, lane="lane_1", roi="roi_1"))
    builder.update(*seen(3, lane="lane_1"))
    context = builder.update(*seen(4, lane="lane_1", roi="roi_1"))

    assert context.vehicles[0].enter_roi_frame == 2
    assert context.vehicles[0].prev_in_roi is True


def test_vehicles_do_not_share_memory():
    builder = RedLightRunningContext()

    builder.update(*seen(1, lane="lane_1", track_id=1))
    builder.update(*seen(2, lane="lane_2", track_id=2))
    context = builder.update(*seen(3, track_id=1))

    assert vehicle(context, 1).last_lane == "lane_1"


def test_a_vehicle_never_in_a_lane_or_the_box_carries_nothing():
    builder = RedLightRunningContext()

    context = builder.update(*seen(1))

    assert vehicle(context).last_lane is None
    assert vehicle(context).in_roi is False
    assert vehicle(context).enter_roi_frame == NEVER


# --- the crossing context -----------------------------------------------------


def crossing(vehicles=(), pedestrians=()) -> CrossingObservation:
    return CrossingObservation(
        vehicles=tuple(Occupant(id, roi) for id, roi in vehicles),
        pedestrians=tuple(Occupant(id, roi) for id, roi in pedestrians),
    )


def test_everyone_in_one_crossing_is_grouped_under_it():
    builder = PedestrianRightOfWayContext()

    context = builder.update(
        crossing(vehicles=[(1, "roi_1")], pedestrians=[(2, "roi_1")]), 0
    )

    assert list(context.rois) == ["roi_1"]
    assert [v.track_id for v in context.rois["roi_1"].vehicles] == [1]
    assert [p.track_id for p in context.rois["roi_1"].pedestrians] == [2]


def test_two_crossings_are_kept_apart():
    # The reason this is grouped at all: a pedestrian at one crossing must not make a
    # violator of a vehicle entering another.
    builder = PedestrianRightOfWayContext()

    context = builder.update(
        crossing(vehicles=[(1, "roi_1")], pedestrians=[(2, "roi_2")]), 0
    )

    assert context.rois["roi_1"].pedestrians == ()
    assert context.rois["roi_2"].vehicles == ()


def test_anyone_outside_every_crossing_is_left_out():
    builder = PedestrianRightOfWayContext()

    context = builder.update(
        crossing(vehicles=[(1, None)], pedestrians=[(2, None)]), 0
    )

    assert context.rois == {}


def test_entering_a_crossing_is_not_yet_a_repeat():
    builder = PedestrianRightOfWayContext()

    builder.update(crossing(vehicles=[(1, None)]), 0)
    context = builder.update(crossing(vehicles=[(1, "roi_1")]), 1)

    assert context.rois["roi_1"].vehicles[0].prev_in_roi is False


def test_staying_in_a_crossing_is_a_repeat_from_the_next_frame_on():
    builder = PedestrianRightOfWayContext()

    builder.update(crossing(vehicles=[(1, None)]), 0)
    builder.update(crossing(vehicles=[(1, "roi_1")]), 1)
    context = builder.update(crossing(vehicles=[(1, "roi_1")]), 2)

    assert context.rois["roi_1"].vehicles[0].prev_in_roi is True


def test_a_vehicle_first_seen_already_inside_can_never_be_reported():
    # Preserved from the source, and the same decision the red-light context makes: we
    # never watched it enter, so we cannot say it entered while somebody was crossing.
    builder = PedestrianRightOfWayContext()

    context = builder.update(crossing(vehicles=[(1, "roi_1")]), 0)

    assert context.rois["roi_1"].vehicles[0].prev_in_roi is True


def test_a_vehicle_outside_a_crossing_is_still_remembered():
    # Left out of the grouping but not out of the cache. Dropping it would make its
    # arrival at the next frame look like a first sighting, which can never be reported.
    builder = PedestrianRightOfWayContext()

    builder.update(crossing(vehicles=[(1, None)]), 0)
    context = builder.update(crossing(vehicles=[(1, "roi_1")]), 1)

    assert context.rois["roi_1"].vehicles[0].prev_in_roi is False
