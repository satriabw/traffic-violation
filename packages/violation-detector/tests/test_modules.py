import numpy as np

from violation_detector.modules import (
    PEDESTRIAN_RIGHT_OF_WAY,
    RED_LIGHT_RUNNING,
    PedestrianRightOfWayModule,
    RedLightRunningModule,
)
from violation_detector.objects import TrackedObject
from violation_detector.regions import Configuration
from violation_detector.traffic_light import GREEN, RED

# A junction: one approach lane, the box past the stop line above it, and the light
# that governs the lane. Small, but a whole scene — the module is built from the same
# kind of document a real site ships.
DOCUMENT = {
    "version": 1,
    "violations": ["rlr_violation"],
    "regions": {
        "lanes": [{"id": "lane_1", "points": [[100, 200], [200, 200], [200, 400], [100, 400]]}],
        "rois": [{"id": "roi_1", "points": [[100, 100], [200, 100], [200, 200], [100, 200]]}],
        "traffic_lights": [
            {
                "id": "tl_1",
                "points": [[10, 10], [25, 10], [25, 55], [10, 55]],
                "controls": ["lane_1"],
            }
        ],
    },
}

# Where the light's polygon lands once reduced to a rectangle to crop.
LIGHT = (slice(10, 56), slice(10, 26))
LIT = {RED: (0, 0, 255), GREEN: (0, 255, 0)}

IN_LANE = (140.0, 180.0, 160.0, 220.0)  # meets the road at (150, 220)
IN_BOX = (140.0, 120.0, 160.0, 160.0)  # meets the road at (150, 160)


def frame(state: str) -> np.ndarray:
    """A frame showing nothing but the light, in the given state."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    lamps = frame[LIGHT]
    lamps[:] = (18, 18, 18)
    height = lamps.shape[0]
    third = height // 3
    band = {RED: slice(0, third), GREEN: slice(2 * third, height)}[state]
    lamps[band] = LIT[state]
    return frame


def car(bbox, track_id: int = 7) -> TrackedObject:
    return TrackedObject(track_id=track_id, bbox=bbox, class_name="car")


def module() -> RedLightRunningModule:
    return RedLightRunningModule(Configuration.from_document(DOCUMENT).regions)


def test_a_vehicle_crossing_on_red_is_reported_once():
    detector = module()
    reported = []

    # Waiting at the line while the light is green, then red, then across it.
    for index, (state, bbox) in enumerate(
        [
            (GREEN, IN_LANE),
            (GREEN, IN_LANE),
            (RED, IN_LANE),
            (RED, IN_BOX),
            (RED, IN_BOX),
            (RED, IN_BOX),
        ]
    ):
        reported.append(detector.detect(frame(state), [car(bbox)], index))

    # Frame 3 is the crossing. Nothing before it, and nothing on the frames it spends
    # sitting in the box afterwards.
    assert [len(violations) for violations in reported] == [0, 0, 0, 1, 0, 0]

    violation = reported[3][0]
    assert violation.type == RED_LIGHT_RUNNING == "red_light_running"
    assert violation.track_id == 7
    assert violation.frame_index == 3
    assert violation.confidence is None


def test_a_vehicle_crossing_on_green_is_never_reported():
    detector = module()

    for index, bbox in enumerate([IN_LANE, IN_LANE, IN_BOX, IN_BOX]):
        assert detector.detect(frame(GREEN), [car(bbox)], index) == []


def test_a_vehicle_that_crossed_on_green_is_not_reported_when_the_light_changes():
    # In the box on green, still in it when the light turns red. Caught by the change,
    # not running it.
    detector = module()

    detector.detect(frame(GREEN), [car(IN_LANE)], 0)
    detector.detect(frame(GREEN), [car(IN_BOX)], 1)

    assert detector.detect(frame(RED), [car(IN_BOX)], 2) == []
    assert detector.detect(frame(RED), [car(IN_BOX)], 3) == []


def test_two_vehicles_are_judged_separately():
    detector = module()

    detector.detect(frame(RED), [car(IN_LANE, 1), car(IN_LANE, 2)], 0)
    violations = detector.detect(frame(RED), [car(IN_BOX, 1), car(IN_LANE, 2)], 1)

    assert [violation.track_id for violation in violations] == [1]


def test_a_pedestrian_in_the_box_on_red_is_not_a_red_light_runner():
    detector = module()
    walker = TrackedObject(track_id=3, bbox=IN_LANE, class_name="person")

    detector.detect(frame(RED), [walker], 0)

    assert detector.detect(frame(RED), [TrackedObject(3, IN_BOX, "person")], 1) == []


def test_a_motorbike_can_run_a_red_light():
    # It could not in the source pipeline: motorbike was in its detection class map
    # but not in its VEHICLES set, so it was annotated as a vehicle and then failed
    # every membership test downstream.
    def rider(bbox):
        return TrackedObject(track_id=4, bbox=bbox, class_name="motorbike")

    detector = module()
    detector.detect(frame(RED), [rider(IN_LANE)], 0)

    assert len(detector.detect(frame(RED), [rider(IN_BOX)], 1)) == 1


def test_a_rule_module_holds_nothing_back_at_the_end():
    # Empty for a rule, which decides on the frame it is given. The hook exists for a
    # module that works on a clip and is always holding a partial one.
    detector = module()

    detector.detect(frame(RED), [car(IN_LANE)], 0)

    assert detector.finish() == []


def test_an_empty_frame_is_not_an_error():
    # Between vehicles, which is most of a video.
    assert module().detect(frame(RED), [], 0) == []


def test_a_scene_that_can_never_fire_builds_anyway():
    # No light, so nothing can ever be reported. Left buildable on purpose — the
    # source pipeline builds it too, and a document that runs today has to keep
    # running.
    document = {**DOCUMENT, "regions": {**DOCUMENT["regions"], "traffic_lights": []}}
    detector = RedLightRunningModule(Configuration.from_document(document).regions)

    assert detector.detect(frame(RED), [car(IN_LANE)], 0) == []
    assert detector.detect(frame(RED), [car(IN_BOX)], 1) == []


# --- pedestrian right of way --------------------------------------------------

BLANK = np.zeros((400, 400, 3), dtype=np.uint8)
ON_KERB = (140.0, 180.0, 160.0, 220.0)  # meets the road at (150, 220), outside roi_1


def crossing_module() -> PedestrianRightOfWayModule:
    return PedestrianRightOfWayModule(Configuration.from_document(DOCUMENT).regions)


def walker(bbox, track_id: int = 2) -> TrackedObject:
    return TrackedObject(track_id=track_id, bbox=bbox, class_name="person")


def test_driving_into_an_occupied_crossing_is_reported_once():
    # Scenario A of the audit: without prev_in_roi this is one report per frame, and a
    # three-second dwell at 30fps inflates the count ninety-fold.
    detector = crossing_module()
    reported = []

    for index, car_bbox in enumerate([ON_KERB, IN_BOX, IN_BOX, IN_BOX]):
        reported.append(
            detector.detect(BLANK, [car(car_bbox), walker(IN_BOX)], index)
        )

    assert [len(violations) for violations in reported] == [0, 1, 0, 0]

    violation = reported[1][0]
    assert violation.type == PEDESTRIAN_RIGHT_OF_WAY == "pedestrian_right_of_way"
    assert violation.track_id == 7
    assert violation.frame_index == 1


def test_a_vehicle_already_stopped_when_somebody_steps_out_is_not_reported():
    # Scenario B, and the case the whole prev_in_roi mechanism exists for. A car queued
    # in the box before anybody left the kerb has not failed to give way to them.
    detector = crossing_module()

    detector.detect(BLANK, [car(ON_KERB)], 0)
    detector.detect(BLANK, [car(IN_BOX)], 1)

    assert detector.detect(BLANK, [car(IN_BOX), walker(IN_BOX)], 2) == []
    assert detector.detect(BLANK, [car(IN_BOX), walker(IN_BOX)], 3) == []


def test_a_vehicle_first_seen_already_inside_is_not_reported():
    # Scenario C. We never watched it enter, so we cannot say it entered while somebody
    # was crossing. At a chunk boundary this is what the LLD's overlap covers.
    detector = crossing_module()

    assert detector.detect(BLANK, [car(IN_BOX), walker(IN_BOX)], 0) == []


def test_an_empty_crossing_reports_nobody():
    detector = crossing_module()

    detector.detect(BLANK, [car(ON_KERB)], 0)

    assert detector.detect(BLANK, [car(IN_BOX)], 1) == []


def test_a_motorbike_alone_in_a_crossing_makes_nobody_a_violator():
    # The deliberate change, end to end. Measured against the source: a car entering a
    # crossing occupied only by a motorbike, with no person anywhere, was reported.
    detector = crossing_module()
    rider = TrackedObject(track_id=4, bbox=IN_BOX, class_name="motorbike")

    detector.detect(BLANK, [car(ON_KERB), rider], 0)

    assert detector.detect(BLANK, [car(IN_BOX), rider], 1) == []


def test_a_cyclist_in_a_crossing_has_right_of_way():
    detector = crossing_module()
    cyclist = TrackedObject(track_id=3, bbox=IN_BOX, class_name="bicycle")

    detector.detect(BLANK, [car(ON_KERB), cyclist], 0)

    assert len(detector.detect(BLANK, [car(IN_BOX), cyclist], 1)) == 1


def test_a_pedestrian_at_another_crossing_makes_nobody_a_violator():
    # Co-presence in one crossing. The document below has a second ROI well away from
    # the first, and somebody standing in it is nothing to do with this vehicle.
    rois = [
        *DOCUMENT["regions"]["rois"],
        {"id": "roi_2", "points": [[300, 100], [380, 100], [380, 180], [300, 180]]},
    ]
    document = {**DOCUMENT, "regions": {**DOCUMENT["regions"], "rois": rois}}
    detector = PedestrianRightOfWayModule(Configuration.from_document(document).regions)
    far_away = walker((330.0, 140.0, 350.0, 170.0))

    detector.detect(BLANK, [car(ON_KERB), far_away], 0)

    assert detector.detect(BLANK, [car(IN_BOX), far_away], 1) == []


def test_every_vehicle_that_pushed_in_is_reported():
    detector = crossing_module()
    first, second = car(ON_KERB, 7), car(ON_KERB, 8)

    detector.detect(BLANK, [first, second, walker(IN_BOX)], 0)
    violations = detector.detect(
        BLANK, [car(IN_BOX, 7), car(IN_BOX, 8), walker(IN_BOX)], 1
    )

    assert sorted(violation.track_id for violation in violations) == [7, 8]


def test_a_rule_module_holds_nothing_back_at_the_end():
    assert crossing_module().finish() == []
