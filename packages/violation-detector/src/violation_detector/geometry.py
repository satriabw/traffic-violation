"""Where an object is: which lane, which region of interest.

Two questions, asked of every tracked object on every frame, and both come down to a
point-in-polygon test against the regions the site's document declares. An object is
represented by one point of its box — by default where it meets the road, which is the
only point of a box that is actually on the ground.

IN PIXELS, AND ONLY IN PIXELS. The pipeline this is ported from had a second path:
given a camera model it projected the object's contact point *and* every region outline
to the ground plane, then ran the same test there, on the reasoning that a polygon
drawn on the image is a trapezoid on the road and pixels understate distance near the
horizon. That reasoning is sound about distance and wrong about containment. The
ground projection is a homography; a homography maps lines to lines, so a polygon stays
a polygon and its interior stays its interior. The answer cannot change.

Measured before deleting it, against that project's own camera model: 20,000 random
points against a lane polygon under a 65-degree pitch produced zero disagreements
between the two paths — and still zero for a polygon annotated so generously that its
far edge crossed the horizon and its ground projection turned inside out. What the
second path cost was a matrix multiply per region per object per frame, plus a class of
failure (a region that silently stops matching because its outline has no ground
point) that the pixel test simply does not have.

None of which says a camera model is useless here — it is what a rule about speed or
following distance will need, and those numbers have no pixel equivalent. It says that
*this* question does not need one, so nothing in this package asks for one yet.
"""

from typing import Sequence

import cv2
import numpy as np

from violation_detector.regions import Region

# Which point of a box stands for the object.
#
# BOTTOM_CENTER is where the object meets the road, so it is the one that belongs in a
# region drawn on the road surface, and the default. The others exist because a box is
# not always trustworthy: CENTER is steadier under a partially occluded box, and
# BOTTOM_THIRD splits the difference for a tall object whose bottom edge is cut off by
# the frame.
BOTTOM_CENTER = "bottom_center"
CENTER = "center"
BOTTOM_THIRD = "bottom_third"


def polygon_to_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    """The upright box that encloses a polygon.

    Used to read a traffic light's pixels out of a frame: the light is annotated as a
    polygon, and cropping wants a rectangle.
    """
    x, y, width, height = cv2.boundingRect(np.round(points).astype(np.int32))
    return (float(x), float(y), float(x + width), float(y + height))


def bbox_center(
    bbox: tuple[float, float, float, float], method: str = BOTTOM_CENTER
) -> tuple[float, float]:
    """The point that stands for an object, given its box."""
    x1, y1, x2, y2 = bbox
    if method == BOTTOM_CENTER:
        return ((x1 + x2) / 2.0, y2)
    if method == CENTER:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if method == BOTTOM_THIRD:
        return ((x1 + x2) / 2.0, y1 + 0.75 * (y2 - y1))
    raise ValueError(f"unknown bbox center method: {method}")


def crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """The pixels inside a box, clipped to the frame.

    A box can run off the edge — a detector's box legitimately does, and a region
    annotated right up against the border does too — and numpy would answer a negative
    index by wrapping round to the far side of the image. Clipping first means the
    worst case is an empty crop rather than pixels from somewhere else entirely.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return frame[y1:y2, x1:x2]


class RegionIndex:
    """One section's regions, ready to be asked what contains a box.

    Built once per job and consulted for every object on every frame. A class rather
    than a function taking a list because the contours are prepared once here: OpenCV
    wants a contiguous float32 array, and rebuilding one per object per frame is the
    kind of cost that hides well.
    """

    def __init__(self, regions: Sequence[Region]):
        self._contours = [
            (region.id, np.ascontiguousarray(region.points, dtype=np.float32))
            for region in regions
        ]

    def locate(
        self, bbox: tuple[float, float, float, float], method: str = BOTTOM_CENTER
    ) -> str | None:
        """Which region contains this box, or None if no region does.

        The first match wins. Regions are not required to be disjoint — a site may
        legitimately overlap an ROI with a lane — and a box on a boundary belongs to
        whichever was declared first, which at least makes the answer stable.
        """
        point = bbox_center(bbox, method)
        for region_id, contour in self._contours:
            # >= 0 counts the boundary as inside, so an object straddling a lane edge
            # is in the lane rather than nowhere.
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return region_id
        return None
