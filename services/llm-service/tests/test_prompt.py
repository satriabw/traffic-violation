"""What the clerk's note is built from.

Two things are worth testing hard here, and they are the two the prompt study showed go
wrong. The first is that nothing countable is left to the model: a study over this exact
data asked two independent arms to screen twenty-six tracks for impossible speeds and got
six and seven against a true nine, so every count in the prompt is computed and every
count is pinned by a test. The second is the register — a clerk cannot act on a track id,
and the way that is guaranteed is by never putting one in the prompt at all.
"""

from datetime import datetime, timezone

from shared.models.detection import ViolationType
from shared.models.explanation import ExplainRequest
from shared.models.violation import EvidenceStatus, TrackSummary, ViolationMetadata

from llm_service.prompt import build_prompt


def _track(track_id, first, last, *, speed=6.0, travel=0.0, width=200.0):
    """One object's window, described by the three things the prompt actually reads.

    `travel` is per-frame drift of the box, which is how the scene summary tells traffic
    apart from a signal head bolted to a pole.
    """
    count = last - first + 1
    return TrackSummary(
        track_id=track_id,
        frame_idxs=list(range(first, last + 1)),
        bboxes=[
            (travel * i, 100.0, travel * i + width, 250.0) for i in range(count)
        ],
        speed=[speed] * count,
        trajectory=[None] * count,
    )


def _request(**overrides):
    return ExplainRequest(
        **{
            "violation_type": ViolationType.RED_LIGHT_RUNNING,
            "detected_at": datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
            "site_name": "Junction 5",
            "frame_index": 159,
            "fps": 60.0,
            **overrides,
        }
    )


def test_no_identifier_ever_reaches_the_prompt():
    """The whole reframing, in one assertion.

    Forbidding the model to mention a track id is the weaker version of this and was what
    the first design did. A vocabulary the model is never given is one it cannot leak, so
    the guarantee lives here rather than in an instruction the model may or may not follow.

    The ids are deliberately outlandish: any of them appearing anywhere in the text is
    then unambiguous rather than a coincidence with a count.
    """
    prompt = build_prompt(
        _request(
            frame_index=4242,
            metadata=ViolationMetadata(
                vehicles=[_track(8675309, 4000, 4242, travel=5.0)],
                pedestrians=[_track(1234567, 4100, 4242, travel=2.0)],
                violator_track_id=8675309,
            ),
        )
    )

    assert "8675309" not in prompt
    assert "1234567" not in prompt
    assert "4242" not in prompt
    for field in ("track_id", "frame_idxs", "bboxes", "violator_track_id", "frame_index"):
        assert field not in prompt


def test_the_moment_is_given_as_a_time_not_a_frame():
    prompt = build_prompt(_request(frame_index=159, fps=60.0))

    assert "2.6 seconds into the footage" in prompt
    assert "159" not in prompt


def test_footage_with_no_known_frame_rate_is_placed_without_a_number():
    """Better to place the event vaguely than to place it in units nobody can use.

    A clerk handed a frame index has been given a number they would have to ask an
    engineer to interpret, which is worse than being told only that it is the moment the
    vehicle was flagged.
    """
    prompt = build_prompt(_request(frame_index=159, fps=None))

    assert "at the moment the vehicle was flagged" in prompt
    assert "159" not in prompt
    assert "seconds into the footage" not in prompt


def test_impossible_speeds_are_counted_and_reported_as_an_equipment_fault():
    """The regression test for the error both study arms made.

    Nine of twenty-six, and the count is the thing under test — one arm said six and the
    other seven, working from the same rows. Several failing is also a different claim
    from one failing, and the prompt has to make that claim rather than leave it to be
    inferred.
    """
    vehicles = [_track(i, 0, 159, speed=8.0, travel=5.0) for i in range(17)]
    vehicles += [_track(100 + i, 0, 159, speed=90.0, travel=5.0) for i in range(9)]
    prompt = build_prompt(
        _request(
            calibration_id="cal-1",
            metadata=ViolationMetadata(vehicles=vehicles, violator_track_id=0),
        )
    )

    assert "9 of 26 objects tracked in this clip" in prompt
    assert "324 km/h" in prompt
    assert "distance calibration is faulty" in prompt
    assert "not several drivers speeding" in prompt


def test_a_pinned_calibration_does_not_vouch_for_impossible_speeds():
    """The guard the earlier rounds got backwards.

    Their version fired only when no calibration was pinned. The one violation in the
    real database carrying 331,529 km/h has a calibration pinned, and the prompt was
    telling the model those numbers were therefore usable.
    """
    prompt = build_prompt(
        _request(
            calibration_id="cal-1",
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, speed=90.0, travel=5.0)],
                violator_track_id=1,
            ),
        )
    )

    assert "SPEED AND DISTANCE: unusable" in prompt


def test_plausible_speeds_under_a_calibration_are_left_usable():
    prompt = build_prompt(
        _request(
            calibration_id="cal-1",
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, speed=8.0, travel=5.0)],
                violator_track_id=1,
            ),
        )
    )

    assert "SPEED AND DISTANCE: available" in prompt


def test_presence_is_membership_of_the_moment_not_a_span_around_it():
    """A track whose samples straddle the violation frame without containing it.

    Asking whether the moment falls between a track's first and last sample is a weaker
    question that this data answers wrongly: the evidence window ends on the violation
    frame, so everything still on screen passes it. Here the second vehicle has a hole
    exactly where the rule fired, and was not there.
    """
    absent = TrackSummary(
        track_id=2,
        frame_idxs=[100, 101, 200, 201],
        bboxes=[(0.0, 0.0, 200.0, 150.0), (50.0, 0.0, 250.0, 150.0)] * 2,
        speed=[6.0] * 4,
    )
    prompt = build_prompt(
        _request(
            frame_index=159,
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0), absent], violator_track_id=1
            ),
        )
    )

    assert "No other vehicle was moving through at that moment." in prompt


def test_objects_that_never_move_are_separated_from_traffic():
    """Signal heads get tracked and filed as vehicles, and the count is what severity reads.

    Eleven of twenty-six in the live row. A clerk told "26 vehicles" pictures a junction
    that was never there.
    """
    vehicles = [_track(1, 0, 159, travel=5.0)]
    vehicles += [_track(10 + i, 0, 159, travel=0.0) for i in range(3)]
    prompt = build_prompt(
        _request(metadata=ViolationMetadata(vehicles=vehicles, violator_track_id=1))
    )

    assert "Of 4 objects the camera counted as vehicles in this clip, 3 never move" in prompt
    assert "overstates how busy the junction was" in prompt


def test_somebody_on_foot_at_the_moment_reaches_the_model():
    # The HIGH severity band turns on this, and it is the one thing the pedestrian rule's
    # own record can genuinely corroborate.
    prompt = build_prompt(
        _request(
            violation_type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY,
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0)],
                pedestrians=[_track(7, 120, 180, travel=2.0)],
                violator_track_id=1,
            ),
        )
    )

    assert "1 person was on foot in the area at that moment." in prompt


def test_a_pedestrian_who_had_already_gone_is_reported_as_gone():
    """The spurious HIGH the old rubric invited.

    Both people in the live row were on the scene and neither was there when the vehicle
    crossed. "On the scene during the violation" grades that HIGH; this does not.
    """
    prompt = build_prompt(
        _request(
            frame_index=159,
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0)],
                pedestrians=[_track(7, 75, 78, travel=2.0)],
                violator_track_id=1,
            ),
        )
    )

    assert "Nobody on foot was there at that moment." in prompt
    assert "1.4 seconds before the vehicle crossed" in prompt


def test_a_detection_too_brief_to_be_a_person_is_flagged_as_such():
    # Four frames at 60fps. "Two pedestrians were present" and "two detections lasting a
    # fifteenth of a second" are different facts about the same record.
    prompt = build_prompt(
        _request(
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0)],
                pedestrians=[_track(7, 75, 78, travel=2.0)],
                violator_track_id=1,
            ),
        )
    )

    assert "camera flickering rather than a real person" in prompt


def test_a_record_that_names_no_vehicle_says_so_rather_than_guessing_one():
    """No falling back to the first vehicle.

    That guess was safe only while a violation's record held exactly one of them. The
    record is the whole scene now, and picking the first would name the car queued behind
    the offender.
    """
    prompt = build_prompt(
        _request(
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0), _track(2, 0, 159, travel=5.0)],
                violator_track_id=None,
            )
        )
    )

    assert "The record does not say which vehicle was flagged" in prompt
    # Nothing is "other" than a vehicle that was never identified.
    assert "other vehicle" not in prompt


def test_a_violation_with_no_scene_recorded_says_so_rather_than_inventing_one():
    prompt = build_prompt(_request(metadata=None))

    assert "WHO ELSE WAS THERE: nothing was recorded" in prompt
    assert "rather than assuming the road was empty" in prompt


def test_the_two_violation_types_get_different_record_checks():
    # What the record can settle differs by rule, and the pedestrian one is genuinely
    # stronger — co-presence in time is computable, a signal colour is not.
    red = build_prompt(_request(violation_type=ViolationType.RED_LIGHT_RUNNING))
    walk = build_prompt(_request(violation_type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY))

    assert "you cannot confirm the red light" in red
    assert "which crossing" not in red
    assert "does NOT keep is which crossing" in walk
    assert "cannot confirm the red light" not in walk


def test_the_prompt_never_asks_for_a_plate_and_forbids_writing_one():
    prompt = build_prompt(_request())

    assert "never write one down" in prompt
    assert "no plate recognition of any kind" in prompt


def test_reading_the_plate_is_only_suggested_when_there_is_footage_to_read():
    """`manual_read` is advice a clerk cannot act on if no clip was ever cut.

    This is why the request carries the evidence status at all — without it the field can
    only ever come back inconclusive, which makes it decorative.
    """
    ready = build_prompt(_request(evidence_status=EvidenceStatus.READY))
    never = build_prompt(_request(evidence_status=None))

    assert "a clerk can go and look at it" in ready
    assert "there may be nothing for a clerk to open" in never


def test_a_distant_vehicle_is_reported_as_too_small_for_a_plate():
    prompt = build_prompt(
        _request(
            metadata=ViolationMetadata(
                vehicles=[_track(1, 0, 159, travel=5.0, width=120.0)],
                violator_track_id=1,
            )
        )
    )

    assert "too small for a plate to be read" in prompt
    # The measurement behind that judgement is not the clerk's problem, and handing it
    # over invites the model to quote a precision the estimate does not have. Scoped to
    # the plate section: the preamble mentions pixel measurements in order to say the
    # model has not been given any.
    plate = prompt[prompt.index("THE PLATE."):prompt.index("BEFORE YOU ENDORSE IT")]
    assert "px" not in plate
    assert "pixel" not in plate
    assert not any(character.isdigit() for character in plate)


def test_an_uncalibrated_violation_is_told_why_the_fallbacks_do_not_work():
    # Naming the failure without naming its mechanism produced a partial correction that
    # read as a complete one, so both routes are still spelled out — in the clerk's
    # register rather than the schema's.
    prompt = build_prompt(_request(calibration_id=None))

    assert "same broken mapping the speeds came from" in prompt
    assert "toward or away from the camera" in prompt


def test_no_severity_band_can_be_reached_by_a_speed():
    # Every over-graded severity in the study came from a run that believed the
    # telemetry. The bands themselves are checked, not the preamble above them — that one
    # says "never on speed", which is the point rather than a violation of it.
    prompt = build_prompt(_request(calibration_id="cal-1"))

    bands = prompt[prompt.index("HIGH   -"):prompt.index("Name in severity_basis")]
    assert "speed" not in bands.lower()
    assert "never on speed" in prompt


def test_severity_and_evidence_strength_are_asked_as_separate_questions():
    # A serious event can be thinly evidenced and a trivial one established beyond doubt.
    # Conflating them is the mistake the two fields exist to make difficult.
    prompt = build_prompt(_request())

    assert "separate question from how well evidenced it is" in prompt


def test_the_unreachable_evidence_band_is_named_as_unreachable():
    # So a clerk reads MEDIUM as a statement about the pipeline rather than about their
    # case — the detector discards the facts that would earn STRONG.
    prompt = build_prompt(_request())

    assert "currently\n           unreachable" in prompt
    assert "Do\n           not award it" in prompt
