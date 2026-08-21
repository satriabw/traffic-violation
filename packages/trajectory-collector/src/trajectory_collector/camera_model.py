"""Pixels to the ground plane.

A camera flattens three dimensions into two and there is no undoing that in general —
a pixel is a ray, not a point, and every point along it lands in the same place. What
makes this tractable is one assumption: the things being tracked are on the ground.
Fix Z=0 and the ray meets the plane exactly once, so a pixel maps to a point after all.

With Z fixed, the 3x4 projection collapses to a 3x3 homography, because the third
column of the rotation matrix only ever multiplies Z:

    [u v 1]ᵀ ~ K [r₁ r₂ r₃ | t] [X Y 0 1]ᵀ  =  K [r₁ r₂ | t] [X Y 1]ᵀ

Inverting that 3x3 is the whole of `project_to_ground`. The inverse is computed once,
at construction, rather than per frame — it depends on nothing that changes.

Reference: https://github.com/AubreyC/trajectory-extractor, by way of the pipeline this
is ported from.
"""

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class CalibrationInvalid(ValueError):
    """The calibration cannot be used to project anything.

    Missing fields, wrong shapes, or a camera whose ground plane is degenerate. Raised
    at construction rather than on some frame in the middle of a video: a calibration
    is checked once and used tens of thousands of times, and a run that produced
    plausible-looking positions from a broken one is far worse than a run that refused
    to start.
    """


# What a calibration document has to contain, and the shape each field has to be.
# Not the OpenCV FileStorage layout ({rows, cols, data}) — this is JSON, and nested
# lists are what JSON callers already have.
REQUIRED = {"camera_matrix": (3, 3), "rot_matrix": (3, 3), "tvec": (3,)}


def _matrix(document: Mapping[str, Any], field: str, shape: tuple[int, ...]) -> np.ndarray:
    if field not in document:
        raise CalibrationInvalid(f"calibration has no {field}")
    try:
        value = np.asarray(document[field], dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationInvalid(f"{field} is not numeric: {error}") from error

    # tvec is written either way in the wild — [x, y, z] or [[x], [y], [z]] — and both
    # mean the same thing, so reshape rather than reject.
    if value.size == int(np.prod(shape)):
        value = value.reshape(shape)
    if value.shape != shape:
        raise CalibrationInvalid(
            f"{field} has shape {value.shape}, expected {shape}"
        )
    return value


def _read(source: str | PathLike | Mapping[str, Any]) -> Mapping[str, Any]:
    """A calibration document, from wherever the caller keeps one.

    A mapping passes straight through: a caller that fetched its calibration from
    object storage already has it parsed, and should not have to write it to a temp
    file to be allowed to use it.

    A path is read as JSON. OpenCV's `.yml` FileStorage format is the other thing
    calibrations are commonly stored in, and it is deliberately not supported here —
    parsing it needs OpenCV, and a dependency on OpenCV is exactly what this package
    exists not to have. Convert it once with `cv2.FileStorage` and keep the JSON.
    """
    if isinstance(source, Mapping):
        return source

    path = Path(source)
    if path.suffix.lower() in {".yml", ".yaml"}:
        raise CalibrationInvalid(
            f"{path} looks like an OpenCV FileStorage calibration. Reading it needs "
            "OpenCV, which this package does not depend on — convert it to JSON "
            "(camera_matrix, rot_matrix, tvec as nested lists) and pass that instead"
        )
    try:
        return json.loads(path.read_text())
    except OSError as error:
        raise CalibrationInvalid(f"cannot read calibration at {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CalibrationInvalid(f"{path} is not valid JSON: {error}") from error


@dataclass(frozen=True)
class CameraModel:
    """Where the camera is and what it sees, reduced to one matrix.

    Built through `from_calibration`; the constructor is for a caller that already
    holds the arrays.
    """

    # Pixel coordinates to ground coordinates, in metres. The only thing projection
    # needs, and the only thing kept — the intrinsics and extrinsics it was built from
    # have no other use here.
    ground_from_pixel: np.ndarray

    @classmethod
    def from_calibration(
        cls, source: str | PathLike | Mapping[str, Any]
    ) -> "CameraModel":
        document = _read(source)
        camera_matrix = _matrix(document, "camera_matrix", REQUIRED["camera_matrix"])
        rot_matrix = _matrix(document, "rot_matrix", REQUIRED["rot_matrix"])
        tvec = _matrix(document, "tvec", REQUIRED["tvec"])

        # r₃ is dropped, not forgotten: it is the column Z multiplies, and Z is zero on
        # the ground plane.
        pixel_from_ground = camera_matrix @ np.column_stack(
            (rot_matrix[:, 0], rot_matrix[:, 1], tvec)
        )
        try:
            ground_from_pixel = np.linalg.inv(pixel_from_ground)
        except np.linalg.LinAlgError as error:
            # A camera looking straight down its own optical axis at the horizon, or a
            # rotation and translation that put the ground plane through the focal
            # point. Physically it means the ground has no image to invert.
            raise CalibrationInvalid(
                f"this camera has no invertible ground plane: {error}"
            ) from error
        return cls(ground_from_pixel=ground_from_pixel)

    def project_to_ground(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) pixel coordinates in, (N, 2) ground coordinates in metres out.

        One matrix multiply for the whole frame rather than one per object — which is
        why the collector projects every box at once before it looks at any of them
        individually.

        A PIXEL ON OR ABOVE THE HORIZON HAS NO GROUND POINT, and comes back non-finite.
        Its ray runs parallel to the ground or away from it and never meets the plane,
        so there is no answer to return — the scale factor below goes to zero, and
        every metre-per-pixel becomes infinite in the limit. It is not an error worth
        raising for: a single detection box whose bottom edge strays above the horizon
        should cost that box, not the video. Callers must drop non-finite rows rather
        than use them; `PinholeCollector` does.
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if len(pixels) == 0:
            return np.empty((0, 2), dtype=float)

        homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
        ground = homogeneous @ self.ground_from_pixel.T
        # Back from homogeneous coordinates. The scale factor is not a constant — it
        # varies per point, and it is exactly what encodes "further away means more
        # metres per pixel". Zero at the horizon, hence the suppressed warning: the
        # infinity it produces is the documented answer, not a mistake.
        with np.errstate(divide="ignore", invalid="ignore"):
            return ground[:, :2] / ground[:, 2:3]
