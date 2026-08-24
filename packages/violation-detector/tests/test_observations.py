import numpy as np

from violation_detector.objects import TrackedObject
from violation_detector.observations import (
    Occupant,
    PedestrianRightOfWayObserver,
    RedLightRunningObserver,
)
from violation_detector.regions import Region, Regions, TrafficLight
from violation_detector.traffic_light import RED, UNKNOWN


def square(id: str, x: float, y: float, size: float = 100.0) -> Region:
    return Region(
        id=id,
        points=np.array(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=np.float32
        ),
    )


REGIONS = Regions(
    lanes=(square("lane_1", 100, 200),),
    rois=(square("roi_1", 100, 100),),
    traffic_lights=(
        TrafficLight(
            id="tl_1",
            points=np.array([[10, 10], [25, 10], [25, 55], [10, 55]], dtype=np.float32),
            controls=("lane_1",),
        ),
    ),
)


def frame_with_red_light() -> np.ndarray:
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[10:56, 10:26] = (18, 18, 18)
    frame[10:25, 10:26] = (0, 0, 255)
    return frame


def test_a_vehicle_is_placed_against_the_lanes_and_the_box():
    observer = RedLightRunningObserver(REGIONS)

    observation = observer.observe(
        frame_with_red_light(), [TrackedObject(7, (140, 180, 160, 220), "car")]
    )

    assert observation.vehicles[0].track_id == 7
    assert observation.vehicles[0].lane == "lane_1"
    assert observation.vehicles[0].roi is None


def test_a_vehicle_outside_every_region_is_placed_nowhere():
    observer = RedLightRunningObserver(REGIONS)

    observation = observer.observe(
        frame_with_red_light(), [TrackedObject(7, (300, 300, 320, 340), "car")]
    )

    assert observation.vehicles[0].lane is None
    assert observation.vehicles[0].roi is None


def test_only_vehicles_are_located():
    # Locating an object costs a point-in-polygon test against every lane and every
    # ROI in the scene, and a pedestrian cannot run a red light.
    observer = RedLightRunningObserver(REGIONS)

    observation = observer.observe(
        frame_with_red_light(),
        [
            TrackedObject(1, (140, 180, 160, 220), "car"),
            TrackedObject(2, (140, 180, 160, 220), "person"),
            TrackedObject(3, (140, 180, 160, 220), "traffic light"),
        ],
    )

    assert [vehicle.track_id for vehicle in observation.vehicles] == [1]


def test_the_light_is_read_from_its_annotated_rectangle():
    observer = RedLightRunningObserver(REGIONS)

    observation = observer.observe(frame_with_red_light(), [])

    assert len(observation.lights) == 1
    assert observation.lights[0].id == "tl_1"
    assert observation.lights[0].state == RED


def test_every_configured_light_is_reported_even_when_it_cannot_be_read():
    # A light missing from this list would be indistinguishable to a rule from a light
    # that is not red, and the rule would fall silent rather than say so.
    observer = RedLightRunningObserver(REGIONS)

    observation = observer.observe(np.zeros((400, 400, 3), dtype=np.uint8), [])

    assert [light.id for light in observation.lights] == ["tl_1"]
    assert observation.lights[0].state == UNKNOWN


def test_observing_the_same_frame_twice_says_the_same_thing():
    # This stage remembers nothing; everything that depends on an earlier frame lives
    # in the context builder.
    observer = RedLightRunningObserver(REGIONS)
    frame = frame_with_red_light()
    objects = [TrackedObject(7, (140, 120, 160, 160), "car")]

    assert observer.observe(frame, objects) == observer.observe(frame, objects)


# --- the crossing observer ----------------------------------------------------

IN_ROI = (140, 120, 160, 160)  # meets the road at (150, 160), inside roi_1
OUTSIDE = (300, 300, 320, 340)


def crossing():
    return PedestrianRightOfWayObserver(REGIONS)


def test_vehicles_and_people_are_told_apart():
    observation = crossing().observe(
        np.zeros((400, 400, 3), dtype=np.uint8),
        [
            TrackedObject(1, IN_ROI, "car"),
            TrackedObject(2, IN_ROI, "person"),
        ],
    )

    assert observation.vehicles == (Occupant(track_id=1, roi="roi_1"),)
    assert observation.pedestrians == (Occupant(track_id=2, roi="roi_1"),)


def test_a_cyclist_is_somebody_to_give_way_to():
    # What these rules care about is that the thing is vulnerable and has right of way,
    # not what it is riding.
    observation = crossing().observe(
        np.zeros((400, 400, 3), dtype=np.uint8), [TrackedObject(3, IN_ROI, "bicycle")]
    )

    assert [occupant.track_id for occupant in observation.pedestrians] == [3]


def test_a_motorbike_is_a_vehicle_not_somebody_to_give_way_to():
    # The one deliberate behaviour change in the port. In the source a motorbike missed
    # the VEHICLES set, fell through to the else branch, and was counted as a
    # pedestrian — so a motorbike alone in a crossing made a violator of every car that
    # entered it, with no person present at all.
    observation = crossing().observe(
        np.zeros((400, 400, 3), dtype=np.uint8), [TrackedObject(4, IN_ROI, "motorbike")]
    )

    assert [occupant.track_id for occupant in observation.vehicles] == [4]
    assert observation.pedestrians == ()


def test_something_that_is_neither_is_nobodys_business():
    # A traffic light, or a class the model had no name for. The source pipeline could
    # not express this: everything that was not a vehicle was filed as a pedestrian.
    observation = crossing().observe(
        np.zeros((400, 400, 3), dtype=np.uint8),
        [TrackedObject(5, IN_ROI, "traffic_light"), TrackedObject(6, IN_ROI, "9")],
    )

    assert observation.vehicles == ()
    assert observation.pedestrians == ()


def test_an_object_outside_every_crossing_is_placed_nowhere():
    observation = crossing().observe(
        np.zeros((400, 400, 3), dtype=np.uint8), [TrackedObject(1, OUTSIDE, "car")]
    )

    assert observation.vehicles == (Occupant(track_id=1, roi=None),)
