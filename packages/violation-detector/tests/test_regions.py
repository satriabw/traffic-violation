import copy

import numpy as np
import pytest

from violation_detector import Configuration, ConfigurationInvalid

# The document a site publishes, verbatim. Kept whole rather than reduced to the parts
# each test needs: this is the shape the API stores and the worker fetches, and a test
# suite that only ever sees fragments of it stops noticing when the real thing drifts.
SAMPLE = {
    "version": 1,
    "violations": ["rlr_violation"],
    "regions": {
        "lanes": [
            {
                "id": "lane_1",
                "points": [[115, 640], [232, 1069], [807, 1067], [354, 639], [114, 641]],
            }
        ],
        "traffic_lights": [
            {
                "id": "tl_1",
                "points": [[33, 470], [31, 516], [50, 519], [53, 458], [31, 459]],
                "controls": ["lane_1"],
            }
        ],
        "rois": [
            {
                "id": "roi_1",
                "points": [[58, 564], [62, 636], [113, 636], [351, 634], [298, 573], [60, 563]],
            }
        ],
    },
}

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def document(**overrides) -> dict:
    """The sample, with the named top-level keys replaced."""
    return {**copy.deepcopy(SAMPLE), **overrides}


def regions(**sections) -> dict:
    """A document whose regions are exactly the sections given."""
    return document(regions=sections)


def test_parses_the_document_a_site_publishes():
    configuration = Configuration.from_document(SAMPLE)

    assert configuration.version == 1
    assert configuration.violations == ("rlr_violation",)
    assert [lane.id for lane in configuration.regions.lanes] == ["lane_1"]
    assert [roi.id for roi in configuration.regions.rois] == ["roi_1"]
    assert [light.id for light in configuration.regions.traffic_lights] == ["tl_1"]


def test_a_traffic_light_carries_the_lanes_it_controls():
    # The junction's wiring, and the document is the only thing that knows it. Without
    # this there is no way to say which vehicles a given light is responsible for, and
    # red-light running is not expressible at all.
    light = Configuration.from_document(SAMPLE).regions.traffic_lights[0]

    assert light.controls == ("lane_1",)


def test_controlled_lanes_maps_each_light_to_its_lanes():
    assert Configuration.from_document(SAMPLE).regions.controlled_lanes() == {
        "tl_1": ("lane_1",)
    }


def test_points_arrive_as_an_array_the_geometry_can_use():
    lane = Configuration.from_document(SAMPLE).regions.lanes[0]

    assert lane.points.shape == (5, 2)
    assert lane.points.dtype == np.float32
    np.testing.assert_array_equal(lane.points[0], [115, 640])


def test_a_section_the_document_leaves_out_is_empty_rather_than_missing():
    # A site with no traffic lights is an ordinary site. Callers iterate these without
    # checking, so the empty case has to be a tuple rather than None.
    configuration = Configuration.from_document(regions(lanes=[{"id": "l", "points": SQUARE}]))

    assert configuration.regions.lanes != ()
    assert configuration.regions.rois == ()
    assert configuration.regions.traffic_lights == ()


def test_regions_may_be_left_out_entirely():
    with pytest.raises(ConfigurationInvalid, match="configuration.regions is missing"):
        Configuration.from_document({"version": 1, "violations": ["rlr_violation"]})


def test_a_version_this_build_does_not_understand_is_refused():
    # Refused rather than parsed hopefully. A configuration is read once and used for a
    # whole run, so a document whose meaning has changed under us produces a run that
    # looks entirely normal and is entirely wrong.
    with pytest.raises(ConfigurationInvalid, match="version is 2, expected 1"):
        Configuration.from_document(document(version=2))


def test_a_document_that_is_not_an_object_is_refused():
    with pytest.raises(ConfigurationInvalid, match="configuration is list"):
        Configuration.from_document([SAMPLE])


def test_an_unknown_region_section_is_refused():
    # The one place strictness pays. "roi" for "rois" would otherwise drop every region
    # of interest on the site, and the only symptom would be a rule that never fires.
    with pytest.raises(ConfigurationInvalid, match="unknown section 'roi'"):
        Configuration.from_document(regions(roi=[{"id": "roi_1", "points": SQUARE}]))


def test_unknown_keys_beside_the_content_are_left_alone():
    # A document carrying a name, a note or an editor's metadata is not wrong. The
    # strictness above is about sections that silently swallow content; this is not one.
    configuration = Configuration.from_document(document(name="Main St / 3rd Ave"))

    assert configuration.violations == ("rlr_violation",)


def test_violations_must_name_at_least_one_rule():
    with pytest.raises(ConfigurationInvalid, match="violations is empty"):
        Configuration.from_document(document(violations=[]))


def test_a_bare_violation_name_is_not_read_as_a_list_of_letters():
    # "rlr_violation" is iterable, and a parser that only checked for iterability would
    # come back with thirteen one-letter rule names and no error.
    with pytest.raises(ConfigurationInvalid, match="violations is str"):
        Configuration.from_document(document(violations="rlr_violation"))


def test_the_same_rule_twice_is_refused():
    # Not deduplicated: two copies of a module each keep their own caches and would
    # report the same vehicle twice, which reads downstream as two violations.
    with pytest.raises(ConfigurationInvalid, match="'rlr_violation' more than once"):
        Configuration.from_document(document(violations=["rlr_violation", "rlr_violation"]))


def test_a_region_needs_an_id():
    with pytest.raises(ConfigurationInvalid, match=r"lanes\[0\].id is missing"):
        Configuration.from_document(regions(lanes=[{"points": SQUARE}]))


def test_a_blank_id_is_not_an_id():
    with pytest.raises(ConfigurationInvalid, match=r"lanes\[0\].id is str"):
        Configuration.from_document(regions(lanes=[{"id": "  ", "points": SQUARE}]))


def test_a_polygon_needs_three_points_to_enclose_anything():
    # Two points describe a line, and every point-in-polygon test against it answers
    # "outside" — a region that exists in the document and nowhere else.
    with pytest.raises(ConfigurationInvalid, match="has 2 points, at least 3"):
        Configuration.from_document(regions(rois=[{"id": "r", "points": [[0, 0], [1, 1]]}]))


def test_a_polygon_must_be_pairs_of_numbers():
    with pytest.raises(ConfigurationInvalid, match=r"expected \(N, 2\)"):
        Configuration.from_document(regions(rois=[{"id": "r", "points": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]}]))


def test_a_polygon_with_a_coordinate_that_is_not_a_number_is_refused():
    with pytest.raises(ConfigurationInvalid, match="not numeric"):
        Configuration.from_document(regions(rois=[{"id": "r", "points": [[0, 0], [1, "x"], [2, 2]]}]))


def test_a_polygon_with_a_nan_is_refused():
    # json has no NaN, but a document assembled in Python does, and a NaN vertex makes
    # every test against that region silently false.
    with pytest.raises(ConfigurationInvalid, match="not a number"):
        Configuration.from_document(regions(rois=[{"id": "r", "points": [[0, 0], [1, float("nan")], [2, 2]]}]))


def test_two_regions_in_one_section_cannot_share_an_id():
    # Lookups answer with an id, and a duplicate makes the answer ambiguous — a
    # violation recorded against "lane_1" would name two places.
    with pytest.raises(ConfigurationInvalid, match="more than one region with id 'lane_1'"):
        Configuration.from_document(
            regions(lanes=[{"id": "lane_1", "points": SQUARE}, {"id": "lane_1", "points": SQUARE}])
        )


def test_a_lane_and_a_region_of_interest_may_share_an_id():
    # Different sections, looked up in different lists. Nothing is served by forbidding
    # a site from numbering both from one.
    configuration = Configuration.from_document(
        regions(lanes=[{"id": "1", "points": SQUARE}], rois=[{"id": "1", "points": SQUARE}])
    )

    assert configuration.regions.lanes[0].id == configuration.regions.rois[0].id


def test_a_light_cannot_control_a_lane_that_was_never_declared():
    # Checked here because here is the only place that can. A rule handed this mapping
    # cannot tell a lane that does not exist from one it has not seen a vehicle in yet,
    # so a typo would simply never fire and never explain itself.
    with pytest.raises(ConfigurationInvalid, match="names 'lane_l', which is not a declared lane"):
        Configuration.from_document(
            regions(
                lanes=[{"id": "lane_1", "points": SQUARE}],
                traffic_lights=[{"id": "tl_1", "points": SQUARE, "controls": ["lane_l"]}],
            )
        )


def test_a_light_may_control_nothing():
    # Useless for red-light running, but not malformed — and a site part way through
    # being annotated should not fail to parse. The rule that needs the mapping is
    # where that becomes a problem.
    configuration = Configuration.from_document(
        regions(traffic_lights=[{"id": "tl_1", "points": SQUARE}])
    )

    assert configuration.regions.traffic_lights[0].controls == ()


def test_an_error_names_the_path_that_failed():
    # Whoever has to fix this is looking at the json, not at this package.
    with pytest.raises(ConfigurationInvalid) as error:
        Configuration.from_document(
            regions(traffic_lights=[{"id": "tl_1", "points": [[0, 0]]}])
        )

    assert "configuration.regions.traffic_lights[0].points" in str(error.value)
