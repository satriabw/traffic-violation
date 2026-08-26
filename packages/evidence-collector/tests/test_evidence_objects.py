import dataclasses

import pytest

from evidence_collector import FrameEntry, ObjectState


def _state(track_id: int = 1, **overrides) -> ObjectState:
    return ObjectState(
        **{
            "track_id": track_id,
            "bbox": (10.0, 20.0, 30.0, 40.0),
            "class_name": "car",
            **overrides,
        }
    )


def test_an_object_state_holds_what_it_was_given():
    state = _state(track_id=7, position=(12.5, -3.0), speed=8.25)

    assert state.track_id == 7
    assert state.bbox == (10.0, 20.0, 30.0, 40.0)
    assert state.class_name == "car"
    assert state.position == (12.5, -3.0)
    assert state.speed == 8.25


def test_position_and_speed_are_absent_by_default():
    # An uncalibrated run is a normal state, not a missing field to be filled with a
    # zero that would read as "on the ground plane origin, stationary".
    state = _state()

    assert state.position is None
    assert state.speed is None


def test_an_object_state_cannot_be_edited_after_the_fact():
    # The whole point of the record. It is read long after the frame it describes is
    # gone, when nothing is left to check it against.
    state = _state()

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.speed = 30.0


def test_a_class_name_is_carried_verbatim_and_never_judged():
    # This package has no vocabulary of vehicles and pedestrians and is not supposed
    # to grow one. Anything the detector emits is a name it will keep.
    assert _state(class_name="e-scooter").class_name == "e-scooter"
    assert _state(class_name="17").class_name == "17"


def test_a_frame_entry_holds_a_frame_index_and_what_was_on_it():
    entry = FrameEntry(frame_index=900, objects=(_state(1), _state(2)))

    assert entry.frame_index == 900
    assert len(entry.objects) == 2


def test_a_frame_with_nothing_on_it_is_a_frame_worth_recording():
    # Half of what makes a window readable: a crossing that was empty and then was not
    # is a different story from one nobody looked at.
    entry = FrameEntry(frame_index=900)

    assert entry.objects == ()
    assert entry.track_ids() == frozenset()


def test_a_timestamp_is_the_callers_and_is_optional():
    assert FrameEntry(frame_index=900).timestamp is None
    assert FrameEntry(frame_index=900, timestamp=1756200000.0).timestamp == 1756200000.0


def test_objects_handed_in_as_a_list_become_a_tuple():
    # The pipeline this is ported from kept its records in lists and then mutated them
    # while reading, so the buffer's contents changed as a side effect of a save. A
    # caller that still holds the list it passed cannot reach what was recorded.
    handed_in = [_state(1)]
    entry = FrameEntry(frame_index=900, objects=handed_in)

    handed_in.append(_state(2))

    assert entry.objects == (_state(1),)
    assert isinstance(entry.objects, tuple)


def test_a_frame_entry_cannot_be_edited_after_the_fact():
    entry = FrameEntry(frame_index=900)

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.frame_index = 901


def test_track_ids_reports_which_tracks_the_frame_saw():
    entry = FrameEntry(frame_index=900, objects=(_state(7), _state(3), _state(7)))

    assert entry.track_ids() == frozenset({3, 7})
