import pytest

from evidence_collector import EvidenceCollector, FrameBuffer, ObjectState


def _state(track_id: int, **overrides) -> ObjectState:
    return ObjectState(
        **{
            "track_id": track_id,
            "bbox": (10.0, 20.0, 30.0, 40.0),
            "class_name": "car",
            **overrides,
        }
    )


class TestBuilding:
    def test_it_is_sized_in_seconds_of_history(self):
        assert EvidenceCollector.over(seconds=5, fps=30).capacity == 151

    def test_it_can_be_built_from_a_buffer_directly(self):
        collector = EvidenceCollector(FrameBuffer(frames=10))

        assert collector.capacity == 10

    def test_a_frame_rate_that_is_not_positive_is_refused(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            EvidenceCollector.over(seconds=5, fps=0)


class TestObserving:
    def test_a_new_collector_holds_nothing(self):
        assert len(EvidenceCollector.over(seconds=5, fps=30)) == 0

    def test_observing_records_a_frame(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)

        collector.observe(900, [_state(7)])

        assert len(collector) == 1

    def test_frames_with_nothing_on_them_are_recorded_too(self):
        # A window is a duration. Recording only the interesting frames would hand back
        # five entries spanning four minutes while claiming to be five seconds.
        collector = EvidenceCollector.over(seconds=1, fps=30)

        collector.observe(900, [])
        collector.observe(901, [])

        assert len(collector) == 2

    def test_it_stops_growing_at_capacity(self):
        collector = EvidenceCollector(FrameBuffer(frames=3))

        for index in range(900, 910):
            collector.observe(index, [_state(7)])

        assert len(collector) == 3

    def test_a_rewound_frame_index_is_refused(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(901, [_state(7)])

        with pytest.raises(ValueError, match="frame indices must increase"):
            collector.observe(900, [_state(7)])

    def test_objects_may_be_any_iterable(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)

        collector.observe(900, (s for s in [_state(7), _state(12)]))

        assert [w.track_id for w in collector.window_for()] == [7, 12]


class TestReadingAWindow:
    def test_the_window_is_the_lead_up_to_now(self):
        collector = EvidenceCollector(FrameBuffer(frames=3))
        for index in range(900, 906):
            collector.observe(index, [_state(7)])

        (window,) = collector.window_for([7])

        assert window.frame_indices == (903, 904, 905)

    def test_the_frame_just_observed_is_in_the_window(self):
        # A rule reports on a frame the caller has just finished analysing, so the
        # moment itself is recorded before the window is read. This is what the `+ 1`
        # in the buffer sizing is for.
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7)])

        (window,) = collector.window_for([7])

        assert window.frame_indices == (900,)

    def test_five_seconds_of_history_survives_the_sixth(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        for index in range(0, 300):
            collector.observe(index, [_state(7)])

        (window,) = collector.window_for([7])

        assert len(window) == 151
        assert window.frame_indices[0] == 149
        assert window.frame_indices[-1] == 299

    def test_only_the_tracks_asked_for_come_back(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7), _state(12)])

        assert [w.track_id for w in collector.window_for([7])] == [7]

    def test_asking_for_nothing_in_particular_takes_everyone(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7), _state(12)])

        assert [w.track_id for w in collector.window_for()] == [7, 12]

    def test_a_track_that_has_rolled_off_is_gone(self):
        collector = EvidenceCollector(FrameBuffer(frames=2))
        collector.observe(900, [_state(7)])
        collector.observe(901, [_state(12)])
        collector.observe(902, [_state(12)])

        assert collector.window_for([7]) == ()

    def test_reading_does_not_consume(self):
        # Two rules firing on one frame each get their own answer.
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7)])

        first = collector.window_for([7])
        second = collector.window_for([7])

        assert first == second
        assert len(collector) == 1

    def test_a_window_read_earlier_does_not_move(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7)])
        (window,) = collector.window_for([7])

        collector.observe(901, [_state(7)])

        assert window.frame_indices == (900,)

    def test_a_recorded_position_comes_back_untouched(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7, position=(12.5, -3.0), speed=8.25)])

        (window,) = collector.window_for([7])

        assert window.positions == ((12.5, -3.0),)
        assert window.speeds == (8.25,)


class TestClearing:
    def test_clearing_forgets_everything(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(900, [_state(7)])

        collector.clear()

        assert len(collector) == 0
        assert collector.window_for() == ()

    def test_clearing_lets_the_indices_start_over(self):
        collector = EvidenceCollector.over(seconds=5, fps=30)
        collector.observe(901, [_state(7)])

        collector.clear()
        collector.observe(900, [_state(7)])

        assert len(collector) == 1
