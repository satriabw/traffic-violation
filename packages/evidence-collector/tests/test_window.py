import dataclasses

import pytest

from evidence_collector import FrameEntry, ObjectState, TrackWindow, summarize


def _state(track_id: int, **overrides) -> ObjectState:
    return ObjectState(
        **{
            "track_id": track_id,
            "bbox": (10.0, 20.0, 30.0, 40.0),
            "class_name": "car",
            **overrides,
        }
    )


def _entry(frame_index: int, *states: ObjectState, timestamp=None) -> FrameEntry:
    return FrameEntry(frame_index=frame_index, objects=states, timestamp=timestamp)


class TestTheParallelInvariant:
    def test_a_window_holds_one_of_everything_per_frame(self):
        window = TrackWindow(
            track_id=7,
            frame_indices=(900, 901),
            positions=((1.0, 2.0), (1.5, 2.5)),
            speeds=(11.0, 12.0),
            bboxes=((0, 0, 10, 10), (1, 1, 11, 11)),
            class_names=("car", "car"),
            timestamps=(None, None),
        )

        assert len(window) == 2

    def test_fields_of_different_lengths_are_refused(self):
        # Silent otherwise, and it arrives as a box drawn around the wrong second of
        # footage.
        with pytest.raises(ValueError, match="parallel fields must be the same length"):
            TrackWindow(
                track_id=7,
                frame_indices=(900, 901),
                positions=((1.0, 2.0),),
                speeds=(11.0, 12.0),
                bboxes=((0, 0, 10, 10), (1, 1, 11, 11)),
                class_names=("car", "car"),
                timestamps=(None, None),
            )

    def test_an_empty_window_is_consistent(self):
        assert len(TrackWindow(track_id=7)) == 0

    def test_a_window_cannot_be_edited_after_the_fact(self):
        window = TrackWindow(track_id=7)

        with pytest.raises(dataclasses.FrozenInstanceError):
            window.track_id = 8

class TestThePivot:
    def test_one_track_across_several_frames_becomes_one_window(self):
        entries = [
            _entry(900, _state(7, bbox=(0.0, 0.0, 10.0, 10.0))),
            _entry(901, _state(7, bbox=(1.0, 1.0, 11.0, 11.0))),
            _entry(902, _state(7, bbox=(2.0, 2.0, 12.0, 12.0))),
        ]

        (window,) = summarize(entries)

        assert window.track_id == 7
        assert window.frame_indices == (900, 901, 902)
        assert window.bboxes == ((0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0), (2.0, 2.0, 12.0, 12.0))

    def test_several_tracks_on_one_frame_become_several_windows(self):
        (seven, twelve) = summarize([_entry(900, _state(7), _state(12))])

        assert (seven.track_id, twelve.track_id) == (7, 12)

    def test_windows_come_back_ordered_by_track_id(self):
        # Two windows produced from one moment come out the same way twice, rather
        # than in whatever order a dict happened to iterate.
        entries = [_entry(900, _state(12), _state(3), _state(7))]

        assert [w.track_id for w in summarize(entries)] == [3, 7, 12]

    def test_frames_stay_in_the_order_they_were_recorded(self):
        entries = [_entry(900, _state(7)), _entry(903, _state(7)), _entry(906, _state(7))]

        (window,) = summarize(entries)

        assert window.frame_indices == (900, 903, 906)

    def test_a_track_missing_from_a_frame_is_absent_not_padded(self):
        # Detection flickers. A car behind a bus for half a second is not a car that
        # was nowhere, and padding the gap would invent evidence.
        entries = [_entry(900, _state(7)), _entry(901), _entry(902, _state(7))]

        (window,) = summarize(entries)

        assert window.frame_indices == (900, 902)
        assert len(window) == 2

    def test_nothing_recorded_yields_no_windows(self):
        assert summarize([]) == ()
        assert summarize([_entry(900)]) == ()

    def test_timestamps_are_carried_through_per_frame(self):
        entries = [_entry(900, _state(7), timestamp=1.0), _entry(901, _state(7), timestamp=2.0)]

        (window,) = summarize(entries)

        assert window.timestamps == (1.0, 2.0)


class TestSelecting:
    def test_only_the_tracks_asked_for_come_back(self):
        entries = [_entry(900, _state(7), _state(12))]

        assert [w.track_id for w in summarize(entries, [7])] == [7]

    def test_none_takes_every_track_in_the_record(self):
        # What a caller wants when the question is "who else was there", and it is
        # bounded by the window rather than by the run.
        entries = [_entry(900, _state(7)), _entry(901, _state(12))]

        assert [w.track_id for w in summarize(entries, None)] == [7, 12]

    def test_a_track_nobody_recorded_comes_back_as_nothing(self):
        # Not as an empty window: handing one back invites a caller to write it down
        # as evidence that this track did nothing.
        assert summarize([_entry(900, _state(7))], [99]) == ()

    def test_asking_for_nothing_gets_nothing(self):
        assert summarize([_entry(900, _state(7))], []) == ()


class TestPositions:
    def test_a_position_is_carried_exactly_as_it_was_handed_in(self):
        entries = [_entry(900, _state(7, position=(12.5, -3.0), speed=8.25))]

        (window,) = summarize(entries)

        assert window.positions == ((12.5, -3.0),)
        assert window.speeds == (8.25,)

    def test_a_frame_nothing_projected_reports_no_position(self):
        # Not a stand-in derived from the box. Without a camera model there is no
        # ground plane and no honest position to report, which is the same reason
        # NullCollector reports nothing at all.
        entries = [_entry(900, _state(7))]

        (window,) = summarize(entries)

        assert window.positions == (None,)
        assert window.speeds == (None,)

    def test_a_frame_nothing_projected_is_still_part_of_the_record(self):
        # Its box is evidence. An object above the horizon, or one in its filter's
        # first frames, was visibly there.
        entries = [
            _entry(900, _state(7, position=(1.0, 2.0))),
            _entry(901, _state(7, position=None)),
            _entry(902, _state(7, position=(3.0, 4.0))),
        ]

        (window,) = summarize(entries)

        assert window.frame_indices == (900, 901, 902)
        assert window.positions == ((1.0, 2.0), None, (3.0, 4.0))

    def test_a_track_that_was_never_projected_still_has_a_window(self):
        entries = [_entry(900, _state(7)), _entry(901, _state(7))]

        (window,) = summarize(entries)

        assert len(window) == 2
        assert window.positions == (None, None)
        assert window.bboxes == ((10.0, 20.0, 30.0, 40.0),) * 2

    def test_nothing_here_derives_a_position_from_a_box(self):
        # The record recomputing anything is a record that can disagree with what it
        # recorded. The box's own bottom-centre lives in the projecting package.
        entries = [_entry(900, _state(7, bbox=(10.0, 20.0, 30.0, 40.0)))]

        (window,) = summarize(entries)

        assert window.positions == (None,)
        assert (20.0, 40.0) not in window.positions


class TestSpeeds:
    def test_an_unmeasured_speed_is_absent_rather_than_zero(self):
        # Zero is a speed somebody could have measured. None is the absence of one, and
        # a filter that has not produced a velocity yet has not measured anything.
        entries = [_entry(900, _state(7, position=(1.0, 2.0), speed=None))]

        (window,) = summarize(entries)

        assert window.speeds == (None,)


class TestClassNames:
    def test_every_frames_name_is_kept(self):
        entries = [
            _entry(900, _state(7, class_name="car")),
            _entry(901, _state(7, class_name="truck")),
        ]

        (window,) = summarize(entries)

        assert window.class_names == ("car", "truck")

    def test_the_convenient_answer_is_the_one_seen_most_often(self):
        entries = [
            _entry(900, _state(7, class_name="car")),
            _entry(901, _state(7, class_name="truck")),
            _entry(902, _state(7, class_name="car")),
        ]

        (window,) = summarize(entries)

        assert window.class_name == "car"

    def test_a_tie_goes_to_the_most_recent(self):
        entries = [
            _entry(900, _state(7, class_name="car")),
            _entry(901, _state(7, class_name="truck")),
        ]

        (window,) = summarize(entries)

        assert window.class_name == "truck"

    def test_an_empty_window_has_no_name(self):
        assert TrackWindow(track_id=7).class_name is None
