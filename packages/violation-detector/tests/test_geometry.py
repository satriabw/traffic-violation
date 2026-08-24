import numpy as np
import pytest

from violation_detector.geometry import (
    BOTTOM_CENTER,
    BOTTOM_THIRD,
    CENTER,
    RegionIndex,
    bbox_center,
    crop,
    polygon_to_bbox,
)
from violation_detector.regions import Region

# A 100x100 square with its top-left at the origin.
SQUARE = Region(id="square", points=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))


def region(id: str, x: float, y: float, size: float = 100.0) -> Region:
    return Region(
        id=id,
        points=np.array(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=np.float32
        ),
    )


def test_the_default_point_is_where_the_object_meets_the_road():
    # The bottom edge, centred. The only point of a box that is actually on the ground
    # plane the regions are drawn on.
    assert bbox_center((10, 20, 30, 60)) == (20.0, 60.0)
    assert bbox_center((10, 20, 30, 60), BOTTOM_CENTER) == (20.0, 60.0)


def test_the_centre_of_a_box_is_available_for_an_occluded_object():
    assert bbox_center((10, 20, 30, 60), CENTER) == (20.0, 40.0)


def test_the_bottom_third_splits_the_difference():
    assert bbox_center((10, 20, 30, 60), BOTTOM_THIRD) == (20.0, 50.0)


def test_an_unknown_centre_method_is_refused():
    # A typo here would silently locate every object by its top-left corner.
    with pytest.raises(ValueError, match="unknown bbox center method: middle"):
        bbox_center((0, 0, 10, 10), "middle")


def test_a_polygon_reduces_to_the_box_that_encloses_it():
    # How a traffic light annotated as a polygon becomes a rectangle to crop.
    assert polygon_to_bbox(np.array([[33, 470], [31, 516], [50, 519], [53, 458]])) == (
        31.0,
        458.0,
        54.0,
        520.0,
    )


def test_cropping_takes_the_pixels_inside_the_box():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)

    np.testing.assert_array_equal(crop(frame, (2, 1, 4, 3)), [[12, 13], [22, 23]])


def test_a_box_running_off_the_frame_is_clipped_rather_than_wrapped():
    # Negative indices are legal in numpy and mean the far side of the image, so an
    # unclipped crop of a box at the left edge would return pixels from the right edge
    # and look perfectly plausible.
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)

    np.testing.assert_array_equal(crop(frame, (-5, -5, 2, 2)), [[0, 1], [10, 11]])
    assert crop(frame, (8, 8, 40, 40)).shape == (2, 2)


def test_a_box_entirely_outside_the_frame_crops_to_nothing():
    # Empty, not wrapped and not raised: whatever reads a traffic light out of this has
    # to cope with an empty crop anyway, and an exception here would cost the frame.
    frame = np.zeros((10, 10), dtype=np.uint8)

    assert crop(frame, (20, 20, 30, 30)).size == 0


def test_an_object_inside_a_region_is_located_by_its_id():
    index = RegionIndex([SQUARE])

    assert index.locate((40, 10, 60, 50)) == "square"


def test_an_object_outside_every_region_is_located_nowhere():
    # None rather than an exception or a default region: most objects on most frames
    # are in none of them, and that is not a problem to report.
    index = RegionIndex([SQUARE])

    assert index.locate((400, 400, 420, 450)) is None


def test_an_object_on_the_boundary_is_inside():
    # A vehicle straddling a lane edge belongs to the lane rather than to nowhere.
    index = RegionIndex([SQUARE])

    assert index.locate((40, 10, 60, 100)) == "square"


def test_the_first_region_declared_wins_where_two_overlap():
    # Regions are not required to be disjoint — a site may overlap an ROI with a lane —
    # so this is about the answer being stable, not about it being the only one.
    overlapping = [region("first", 0, 0), region("second", 50, 0)]

    assert RegionIndex(overlapping).locate((70, 10, 80, 50)) == "first"
    assert RegionIndex(overlapping[::-1]).locate((70, 10, 80, 50)) == "second"


def test_a_site_with_no_regions_locates_nothing():
    assert RegionIndex([]).locate((0, 0, 10, 10)) is None


def test_which_point_stands_for_the_object_decides_the_answer():
    # A tall box whose centre is inside the region but whose wheels are beyond it. Not
    # a corner case: it is the difference between a vehicle being in the junction and
    # merely overhanging it.
    index = RegionIndex([SQUARE])
    overhanging = (40, 40, 60, 140)

    assert index.locate(overhanging, BOTTOM_CENTER) is None
    assert index.locate(overhanging, CENTER) == "square"
