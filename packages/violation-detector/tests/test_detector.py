import numpy as np
import pytest

import violation_detector
from violation_detector import (
    Configuration,
    ConfigurationInvalid,
    Detector,
    TrackedObject,
    Violation,
    ViolationModule,
    get_detector,
)
from violation_detector.modules import (
    PEDESTRIAN_RIGHT_OF_WAY,
    RED_LIGHT_RUNNING,
    PedestrianRightOfWayModule,
    RedLightRunningModule,
)
from violation_detector.registry import ModuleContext, factory_for, register, registered

# The document from the README, verbatim — the schema a site actually ships.
DOCUMENT = {
    "version": 1,
    "violations": ["rlr_violation"],
    "regions": {
        "lanes": [{"id": "lane_1", "points": [[115, 640], [232, 1069], [807, 1067], [354, 639]]}],
        "traffic_lights": [
            {
                "id": "tl_1",
                "points": [[33, 470], [31, 516], [50, 519], [53, 458]],
                "controls": ["lane_1"],
            }
        ],
        "rois": [{"id": "roi_1", "points": [[58, 564], [62, 636], [113, 636], [351, 634]]}],
    },
}

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def configuration(**overrides) -> Configuration:
    return Configuration.from_document({**DOCUMENT, **overrides})


class Spy(ViolationModule):
    """A rule that reports on demand, so aggregation can be observed."""

    type = "spy"

    def __init__(self, context: ModuleContext, held: int = 0):
        self.context = context
        self.calls: list[int] = []
        self._held = held

    def detect(self, frame, tracked_objects, frame_index):
        self.calls.append(frame_index)
        return [Violation(type=self.type, track_id=1, frame_index=frame_index)]

    def finish(self):
        return [Violation(type=self.type, track_id=9, frame_index=self._held)]


@pytest.fixture
def spy_registered():
    """Registers a second rule for the duration of one test."""
    register("spy_violation", "spy", Spy)
    yield
    del violation_detector.registry._REGISTRY["spy_violation"]


def test_a_detector_is_built_from_a_sites_document():
    detector = get_detector(configuration())

    modules = detector.get_modules()
    assert len(modules) == 1
    assert isinstance(modules[0], RedLightRunningModule)
    assert modules[0].type == RED_LIGHT_RUNNING


def test_a_document_naming_a_rule_this_build_does_not_carry_is_refused():
    # Named at parse time and unresolvable here. Silence would be a job that watches
    # an hour of footage for a rule nobody implemented.
    with pytest.raises(ConfigurationInvalid, match="not a rule this build carries"):
        get_detector(configuration(violations=["speeding"]))


def test_the_error_says_what_would_have_worked():
    with pytest.raises(ConfigurationInvalid, match="rlr_violation"):
        get_detector(configuration(violations=["rlr"]))


def test_asking_for_the_violation_the_site_is_configured_for_builds_it():
    detector = get_detector(configuration(), types=[RED_LIGHT_RUNNING])

    assert len(detector.get_modules()) == 1


def test_asking_only_for_a_violation_this_site_is_not_configured_for_builds_nothing():
    # The site is the authority on what can be watched for. Not an error — a job may
    # legitimately ask for more than a given junction is annotated for.
    detector = get_detector(configuration(), types=["pedestrian_right_of_way"])

    assert detector.get_modules() == ()
    assert detector.detect(FRAME, [], 0) == []


def test_a_rule_the_job_did_not_ask_for_is_not_run():
    # The job is the authority on what was wanted.
    detector = get_detector(configuration(), types=[])

    assert detector.get_modules() == ()


def test_no_types_at_all_runs_everything_the_document_names():
    assert len(get_detector(configuration()).get_modules()) == 1


def test_types_are_canonical_values_not_document_names():
    # The worker passes `[t.value for t in job.types]` and holds no translation table.
    assert get_detector(configuration(), types=["rlr_violation"]).get_modules() == ()


def test_violations_from_every_rule_are_aggregated(spy_registered):
    detector = get_detector(configuration(violations=["rlr_violation", "spy_violation"]))

    violations = detector.detect(FRAME, [], 4)

    assert len(detector.get_modules()) == 2
    assert [violation.type for violation in violations] == ["spy"]


def test_rules_run_in_the_order_the_document_names_them(spy_registered):
    detector = get_detector(configuration(violations=["spy_violation", "rlr_violation"]))

    assert [module.type for module in detector.get_modules()] == ["spy", RED_LIGHT_RUNNING]


def test_finishing_drains_every_rule(spy_registered):
    # A module that buffers reports late, and the frame it names is not the last one.
    detector = get_detector(configuration(violations=["spy_violation"]))

    detector.detect(FRAME, [], 30)
    held = detector.finish()

    assert [(v.track_id, v.frame_index) for v in held] == [(9, 0)]


def test_a_factory_is_given_the_scene_and_the_frame_rate(spy_registered):
    detector = get_detector(configuration(violations=["spy_violation"]), fps=25.0)

    context = detector.get_modules()[0].context
    assert context.fps == 25.0
    assert [lane.id for lane in context.regions.lanes] == ["lane_1"]


def test_the_frame_rate_is_optional(spy_registered):
    detector = get_detector(configuration(violations=["spy_violation"]))

    assert detector.get_modules()[0].context.fps is None


def test_two_detectors_from_one_configuration_share_no_state():
    # Per job, because tracker ids restart at 1 for every video. A module reused
    # across jobs would answer about a completely different vehicle.
    first, second = get_detector(configuration()), get_detector(configuration())

    assert first.get_modules()[0] is not second.get_modules()[0]


def test_a_detector_with_no_rules_is_usable():
    # What the worker holds when a job asks for nothing, the way NullCollector works.
    detector = Detector()

    assert detector.detect(FRAME, [TrackedObject(1, (0, 0, 10, 10), "car")], 0) == []
    assert detector.finish() == []


def test_registering_replaces_an_existing_rule(spy_registered):
    # How a deployment swaps an implementation without editing this package.
    register("spy_violation", "spy", lambda context: Spy(context, held=7))

    detector = get_detector(configuration(violations=["spy_violation"]))

    assert detector.finish()[0].frame_index == 7


def test_the_registry_reports_what_it_carries():
    assert registered() == ("pdx_violation", "rlr_violation")
    assert factory_for("rlr_violation")[0] == RED_LIGHT_RUNNING
    assert factory_for("pdx_violation")[0] == PEDESTRIAN_RIGHT_OF_WAY


def test_a_site_can_run_both_rules_at_once():
    detector = get_detector(configuration(violations=["rlr_violation", "pdx_violation"]))

    assert [type(module) for module in detector.get_modules()] == [
        RedLightRunningModule,
        PedestrianRightOfWayModule,
    ]


def test_a_job_can_ask_for_one_of_the_two_a_site_runs():
    # The intersection doing real work: this junction watches for both, this job wants
    # only the crossing.
    detector = get_detector(
        configuration(violations=["rlr_violation", "pdx_violation"]),
        types=[PEDESTRIAN_RIGHT_OF_WAY],
    )

    assert [module.type for module in detector.get_modules()] == [
        PEDESTRIAN_RIGHT_OF_WAY
    ]


def test_both_rules_report_into_one_list():
    # Nothing deduplicates across them: a vehicle that ran a red light into an occupied
    # crossing broke two rules, and which to keep is the caller's judgement.
    detector = get_detector(configuration(violations=["rlr_violation", "pdx_violation"]))

    assert detector.detect(FRAME, [], 0) == []
    assert detector.finish() == []
