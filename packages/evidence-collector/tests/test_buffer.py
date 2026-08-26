import pytest

from evidence_collector import FrameBuffer, FrameEntry, ObjectState


def _entry(frame_index: int, *track_ids: int) -> FrameEntry:
    return FrameEntry(
        frame_index=frame_index,
        objects=tuple(
            ObjectState(track_id=track_id, bbox=(0.0, 0.0, 1.0, 1.0), class_name="car")
            for track_id in track_ids
        ),
    )


def _indices(buffer: FrameBuffer) -> list[int]:
    return [entry.frame_index for entry in buffer.entries()]


class TestSizing:
    def test_five_seconds_at_thirty_fps_holds_the_lead_up_plus_the_moment(self):
        # 150 frames of history, and the frame the rule fired on. Sizing it to 150
        # would push the oldest frame out to make room for the present one.
        assert FrameBuffer.over(seconds=5, fps=30).capacity == 151

    def test_a_fractional_frame_rate_rounds_to_whole_frames(self):
        # 29.97 * 5 is 149.85 frames, and there is no such thing as .85 of a record.
        assert FrameBuffer.over(seconds=5, fps=29.97).capacity == 151

    def test_a_buffer_can_be_sized_in_frames_directly(self):
        assert FrameBuffer(frames=10).capacity == 10

    @pytest.mark.parametrize("fps", [0, -1, -30.0])
    def test_a_frame_rate_that_is_not_positive_is_refused(self, fps):
        with pytest.raises(ValueError, match="fps must be positive"):
            FrameBuffer.over(seconds=5, fps=fps)

    @pytest.mark.parametrize("seconds", [0, -1, -0.5])
    def test_a_window_that_is_not_positive_is_refused(self, seconds):
        with pytest.raises(ValueError, match="seconds must be positive"):
            FrameBuffer.over(seconds=seconds, fps=30)

    @pytest.mark.parametrize("frames", [0, -1])
    def test_a_buffer_of_nothing_is_refused(self, frames):
        # It would report an empty window for every violation, which reads exactly
        # like a junction where nothing led up to anything.
        with pytest.raises(ValueError, match="frames must be at least 1"):
            FrameBuffer(frames=frames)


class TestRecording:
    def test_a_new_buffer_is_empty(self):
        buffer = FrameBuffer(frames=3)

        assert len(buffer) == 0
        assert buffer.entries() == ()

    def test_an_empty_buffer_is_falsy(self):
        # Defined `__len__`, so `if buffer:` means what it looks like. The pipeline
        # this is ported from omitted it and guarded three call sites with `if not
        # self._frame_buffer`, every one of which was unreachable.
        assert not FrameBuffer(frames=3)

    def test_a_buffer_holding_anything_is_truthy(self):
        buffer = FrameBuffer(frames=3)
        buffer.add(_entry(900))

        assert buffer

    def test_entries_come_back_oldest_first(self):
        # The order the events happened in. A window read backwards is a window
        # somebody eventually reads wrong.
        buffer = FrameBuffer(frames=5)
        buffer.extend([_entry(900), _entry(901), _entry(902)])

        assert _indices(buffer) == [900, 901, 902]

    def test_it_fills_to_capacity_and_stops_growing(self):
        buffer = FrameBuffer(frames=3)

        for index, expected in enumerate([1, 2, 3, 3, 3], start=900):
            buffer.add(_entry(index))
            assert len(buffer) == expected

    def test_the_oldest_frame_is_dropped_once_it_is_full(self):
        buffer = FrameBuffer(frames=3)
        buffer.extend(_entry(index) for index in range(900, 906))

        assert _indices(buffer) == [903, 904, 905]

    def test_what_was_recorded_survives_the_round_trip(self):
        buffer = FrameBuffer(frames=3)
        buffer.add(_entry(900, 7, 12))

        assert buffer.entries()[0].track_ids() == frozenset({7, 12})

    def test_frames_with_nothing_on_them_still_take_a_slot(self):
        # They are evidence too: four seconds of an empty crossing is why the fifth
        # matters. Skipping them would silently stretch the window past its duration.
        buffer = FrameBuffer(frames=3)
        buffer.extend([_entry(900), _entry(901, 7), _entry(902)])

        assert _indices(buffer) == [900, 901, 902]


class TestOrdering:
    def test_a_repeated_frame_index_is_refused(self):
        # The caller fed a frame twice. The window would still be the right length and
        # still look like plausible history, while describing time that never happened.
        buffer = FrameBuffer(frames=5)
        buffer.add(_entry(900))

        with pytest.raises(ValueError, match="frame indices must increase"):
            buffer.add(_entry(900))

    def test_a_rewound_frame_index_is_refused(self):
        buffer = FrameBuffer(frames=5)
        buffer.add(_entry(901))

        with pytest.raises(ValueError, match="frame indices must increase"):
            buffer.add(_entry(900))

    def test_gaps_are_fine(self):
        # A caller sampling every third frame, or reading a chunk that starts at 900.
        # Only going backwards is a mistake; skipping is a decision.
        buffer = FrameBuffer(frames=5)
        buffer.extend([_entry(900), _entry(903), _entry(906)])

        assert _indices(buffer) == [900, 903, 906]

    def test_a_refused_frame_leaves_the_buffer_as_it_was(self):
        buffer = FrameBuffer(frames=5)
        buffer.add(_entry(900))

        with pytest.raises(ValueError):
            buffer.add(_entry(899))

        assert _indices(buffer) == [900]

    def test_the_index_it_has_reached_survives_eviction(self):
        # The guard is about the caller's position in the video, not about what is
        # still held. Frame 900 having rolled off does not make it addable again.
        buffer = FrameBuffer(frames=2)
        buffer.extend([_entry(900), _entry(901), _entry(902)])

        with pytest.raises(ValueError, match="frame indices must increase"):
            buffer.add(_entry(900))


class TestClearing:
    def test_clearing_empties_it(self):
        buffer = FrameBuffer(frames=3)
        buffer.extend([_entry(900), _entry(901)])

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.entries() == ()

    def test_clearing_forgets_how_far_the_indices_had_got(self):
        # The one caller this exists for is a genuine restart, which starts over at an
        # index it has already been past.
        buffer = FrameBuffer(frames=3)
        buffer.add(_entry(901))

        buffer.clear()
        buffer.add(_entry(900))

        assert _indices(buffer) == [900]

    def test_capacity_is_unchanged_by_clearing(self):
        buffer = FrameBuffer(frames=3)
        buffer.add(_entry(900))

        buffer.clear()

        assert buffer.capacity == 3


class TestSnapshots:
    def test_a_window_read_earlier_does_not_move(self):
        # What comes back is a reading taken then. A caller holding one while the
        # buffer keeps filling would find its evidence had changed underneath it.
        buffer = FrameBuffer(frames=5)
        buffer.add(_entry(900))
        window = buffer.entries()

        buffer.add(_entry(901))

        assert _indices(buffer) == [900, 901]
        assert [entry.frame_index for entry in window] == [900]

    def test_a_window_cannot_be_edited_into_the_buffer(self):
        buffer = FrameBuffer(frames=5)
        buffer.add(_entry(900))

        assert isinstance(buffer.entries(), tuple)
