import numpy as np

from violation_detector.objects import TrackedObject
from violation_detector.observations import RedLightRunningObserver
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
