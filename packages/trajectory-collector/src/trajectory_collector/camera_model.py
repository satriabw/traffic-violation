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

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CalibrationInvalid(ValueError):
    """The calibration cannot be used to project anything.

    An unreadable document, missing fields, wrong shapes, or a camera whose ground
    plane is degenerate. Raised at construction rather than on some frame in the middle
    of a video: a calibration is checked once and used tens of thousands of times, and
    a run that produced plausible-looking positions from a broken one is far worse than
    a run that refused to start.
    """


# The nodes a calibration has to carry, and the shape each one has to be. dist_coeffs
# is deliberately not among them: nothing here undistorts, because the only projection
# it performs is the ground-plane homography, and a node that is read but unused
# invites the belief that it is doing something.
REQUIRED = {"camera_matrix": (3, 3), "rot_matrix": (3, 3), "tvec": (3,)}


def _matrix(value: Any, field: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationInvalid(f"{field} is not numeric: {error}") from error

    # tvec is written either way in the wild — [x, y, z] or [[x], [y], [z]], and OpenCV
    # hands back the latter — and both mean the same thing, so reshape rather than
    # reject.
    if array.size == int(np.prod(shape)):
        array = array.reshape(shape)
    if array.shape != shape:
        raise CalibrationInvalid(f"{field} has shape {array.shape}, expected {shape}")
    return array


def _read(source: str | PathLike | bytes | bytearray) -> dict[str, np.ndarray]:
    """An OpenCV FileStorage calibration, parsed.

    One format, because that is the one camera calibration tools write. Bytes are the
    document itself, for a caller that fetched it from object storage and has no file
    to point at; anything else is a path.

    Parsed in memory either way: `FILE_STORAGE_MEMORY` reads a string directly, so a
    document out of object storage never has to touch a disk on its way in.
    """
    if isinstance(source, (bytes, bytearray)):
        raw, described_as = bytes(source), "calibration document"
    else:
        path = Path(source)
        described_as = str(path)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise CalibrationInvalid(
                f"cannot read calibration at {path}: {error}"
            ) from error

    try:
        storage = cv2.FileStorage(
            raw.decode("utf-8", errors="replace"),
            cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY,
        )
    except Exception as error:
        # Not just cv2.error: OpenCV surfaces a malformed document as a bare
        # SystemError out of the constructor, which no narrower clause would catch and
        # which no caller of this package should ever have to see.
        raise CalibrationInvalid(
            f"{described_as} is not a readable OpenCV calibration: {error}"
        ) from error

    try:
        # Absent nodes are left out rather than stored as None, so a document missing
        # one produces the same error as a mapping missing a key would.
        return {
            name: storage.getNode(name).mat()
            for name in REQUIRED
            if not storage.getNode(name).empty()
        }
    except Exception as error:
        raise CalibrationInvalid(
            f"{described_as} could not be read as an OpenCV calibration: {error}"
        ) from error
    finally:
        storage.release()


@dataclass(frozen=True)
class CameraModel:
    """Where the camera is and what it sees, reduced to one matrix.

    Built through `from_calibration`, or `from_matrices` for a caller that already
    holds the arrays.
    """

    # Pixel coordinates to ground coordinates, in metres. The only thing projection
    # needs, and the only thing kept — the intrinsics and extrinsics it was built from
    # have no other use here.
    ground_from_pixel: np.ndarray

    @classmethod
    def from_calibration(cls, source: str | PathLike | bytes | bytearray) -> "CameraModel":
        """The camera an OpenCV FileStorage calibration describes."""
        document = _read(source)
        missing = [name for name in REQUIRED if name not in document]
        if missing:
            raise CalibrationInvalid(f"calibration has no {', '.join(missing)}")
        return cls.from_matrices(**document)

    @classmethod
    def from_matrices(
        cls, camera_matrix: Any, rot_matrix: Any, tvec: Any
    ) -> "CameraModel":
        """The camera these intrinsics and extrinsics describe.

        Separate from `from_calibration` so that having the numbers and having a
        document are different problems. Anything array-like will do, and a matrix may
        be flat and row-major — the form OpenCV stores one in.
        """
        camera_matrix = _matrix(camera_matrix, "camera_matrix", REQUIRED["camera_matrix"])
        rot_matrix = _matrix(rot_matrix, "rot_matrix", REQUIRED["rot_matrix"])
        tvec = _matrix(tvec, "tvec", REQUIRED["tvec"])

        # r₃ is dropped, not forgotten: it is the column Z multiplies, and Z is zero on
        # the ground plane.
        pixel_from_ground = camera_matrix @ np.column_stack(
            (rot_matrix[:, 0], rot_matrix[:, 1], tvec)
        )
        try:
            ground_from_pixel = np.linalg.inv(pixel_from_ground)
        except np.linalg.LinAlgError as error:
            # A rotation and translation that put the ground plane through the focal
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
