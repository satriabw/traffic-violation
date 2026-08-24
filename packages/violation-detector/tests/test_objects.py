import dataclasses

import pytest

from violation_detector import TrackedObject, Violation


def tracked(class_name: str) -> TrackedObject:
    return TrackedObject(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_name=class_name)


@pytest.mark.parametrize("name", ["car", "bus", "truck", "motorbike", "motorcycle"])
def test_the_things_that_are_vehicles(name):
    assert tracked(name).is_vehicle
    assert not tracked(name).is_pedestrian


@pytest.mark.parametrize("name", ["person", "pedestrian", "bicycle", "cyclist", "e-scooter"])
def test_the_things_that_are_pedestrians(name):
    # A cyclist is a person on a bicycle. What these rules care about is that the thing
    # is vulnerable and has right of way, not what it is riding.
    assert tracked(name).is_pedestrian
    assert not tracked(name).is_vehicle


def test_a_motorbike_is_a_vehicle():
    # A regression, and the reason these sets are defined in one place. The pipeline
    # this is ported from had motorbike in its detection class map but not in its
    # VEHICLES set, so the two disagreed — and quietly, in opposite directions. A
    # motorbike could never run a red light, and in the pedestrian rule it fell through
    # to the else branch and counted as a *pedestrian*, which made every car sharing
    # the region a violator.
    assert tracked("motorbike").is_vehicle
    assert not tracked("motorbike").is_pedestrian


def test_an_object_no_rule_has_an_opinion_about_is_neither():
    # Not an error. A detector's label space is wider than these rules care about, and
    # deciding which types matter is the rule's job, not the detector's.
    assert not tracked("traffic_light").is_vehicle
    assert not tracked("fire hydrant").is_pedestrian


def test_a_tracked_object_cannot_be_edited_after_the_fact():
    # A module may hold one of these in a cache for as long as a track lives.
    with pytest.raises(dataclasses.FrozenInstanceError):
        tracked("car").bbox = (1.0, 1.0, 2.0, 2.0)


def test_a_violation_records_what_happened_and_when():
    violation = Violation(type="red_light_running", track_id=7, frame_index=912)

    assert violation.type == "red_light_running"
    assert violation.track_id == 7
    assert violation.frame_index == 912


def test_a_rule_reports_no_confidence():
    # A rule either fired or it did not. 1.0 would be a score nobody computed; a model
    # based module is what fills this in.
    assert Violation(type="red_light_running", track_id=7, frame_index=912).confidence is None


def test_a_violation_cannot_be_edited_after_the_fact():
    violation = Violation(type="red_light_running", track_id=7, frame_index=912)

    with pytest.raises(dataclasses.FrozenInstanceError):
        violation.track_id = 8
