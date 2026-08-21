import numpy as np
import pytest

from trajectory_collector import CalibrationInvalid, CameraModel


def overhead(focal: float = 1000.0, height: float = 100.0) -> dict:
    """A camera looking straight down from `height` metres.

    Deliberately the simplest camera that is still a real one: with no rotation and the
    camera on the optical axis, the ground is a plain scaling of the image at
    `height / focal` metres per pixel. Defaults give 0.1 m/px, so every expected value
    in these tests can be worked out by hand.
    """
    return {
        "camera_matrix": [[focal, 0.0, 0.0], [0.0, focal, 0.0], [0.0, 0.0, 1.0]],
        "rot_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tvec": [0.0, 0.0, height],
    }


def test_a_pixel_becomes_a_point_in_metres():
    model = CameraModel.from_matrices(**overhead())

    ground = model.project_to_ground(np.array([[100.0, 250.0]]))

    # 0.1 metres per pixel, straight down.
    assert ground == pytest.approx(np.array([[10.0, 25.0]]))


def test_the_whole_frame_projects_in_one_call():
    model = CameraModel.from_matrices(**overhead())

    ground = model.project_to_ground(np.array([[0.0, 0.0], [10.0, 20.0], [-30.0, 5.0]]))

    assert ground == pytest.approx(np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]]))


def test_an_empty_frame_projects_to_nothing():
    # Most frames of most footage have something in them, but not all, and an empty
    # array must not become an exception halfway down a video.
    ground = CameraModel.from_matrices(**overhead()).project_to_ground(
        np.empty((0, 2))
    )

    assert ground.shape == (0, 2)


def test_a_ground_point_survives_the_round_trip():
    # The real check, against a camera at an angle rather than the hand-checkable one:
    # project a known ground point into the image, and the model must put it back
    # where it came from.
    document = {
        "camera_matrix": [[5527.9, 0.0, 960.0], [0.0, 5527.9, 544.0], [0.0, 0.0, 1.0]],
        "rot_matrix": [
            [-0.0568334907, 0.99814862, -0.0216619018],
            [-0.0312085897, 0.0199102219, 0.999314547],
            [0.997895777, 0.0574705712, 0.0300192442],
        ],
        "tvec": [-1.62685621, 0.940755188, 91.4862976],
    }
    model = CameraModel.from_matrices(**document)
    pixel_from_ground = np.linalg.inv(model.ground_from_pixel)

    truth = np.array([[3.0, 12.0], [-7.5, 40.0]])
    homogeneous = np.column_stack((truth, np.ones(len(truth)))) @ pixel_from_ground.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]

    assert model.project_to_ground(pixels) == pytest.approx(truth)


def test_further_away_is_more_metres_per_pixel():
    # The property that makes projecting worth doing at all: two objects the same
    # distance apart on screen are not the same distance apart on the ground.
    model = CameraModel.from_matrices(**_tilted())

    # A hundred pixels apart in both cases; the far pair is nearer the horizon.
    near = model.project_to_ground(np.array([[0.0, 500.0], [0.0, 400.0]]))
    far = model.project_to_ground(np.array([[0.0, 200.0], [0.0, 100.0]]))

    near_gap = abs(near[1, 1] - near[0, 1])
    far_gap = abs(far[1, 1] - far[0, 1])
    assert far_gap > near_gap


def _tilted() -> dict:
    """A camera tilted forward, so the top of the image is further away.

    Its horizon works out at v = 0: u = 1000X / (Y + 40), v = 5000 / (Y + 40), so a
    point at v = 0 is one infinitely far down the road.
    """
    return {
        "camera_matrix": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
        "rot_matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        "tvec": [0.0, 5.0, 40.0],
    }


def test_a_pixel_on_the_horizon_has_no_ground_point():
    # Its ray runs parallel to the ground and never meets it, so there is no answer to
    # return. Non-finite rather than an exception: one stray detection box should cost
    # that box, not the video.
    ground = CameraModel.from_matrices(**_tilted()).project_to_ground(
        np.array([[0.0, 0.0]])
    )

    assert not np.isfinite(ground).all()


def test_one_pixel_on_the_horizon_does_not_spoil_the_rest_of_the_frame():
    # The whole frame projects in one multiply, so a single unusable box must not take
    # its neighbours with it.
    ground = CameraModel.from_matrices(**_tilted()).project_to_ground(
        np.array([[0.0, 500.0], [0.0, 0.0], [0.0, 400.0]])
    )

    assert np.isfinite(ground[[0, 2]]).all()
    assert not np.isfinite(ground[1]).all()


# --- documents that cannot be used --------------------------------------------


def test_a_field_of_the_wrong_shape_is_rejected():
    document = overhead()
    document["camera_matrix"] = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(CalibrationInvalid, match="shape"):
        CameraModel.from_matrices(**document)


def test_a_field_that_is_not_numbers_is_rejected():
    document = overhead()
    document["tvec"] = ["over", "there", "somewhere"]

    with pytest.raises(CalibrationInvalid, match="not numeric"):
        CameraModel.from_matrices(**document)


def test_a_camera_with_no_invertible_ground_plane_is_rejected():
    # Physically: the ground has no image to invert. Numerically: a singular matrix,
    # which would otherwise surface as a LinAlgError from the first frame of a video.
    document = overhead()
    document["tvec"] = [0.0, 0.0, 0.0]

    with pytest.raises(CalibrationInvalid, match="ground plane"):
        CameraModel.from_matrices(**document)


def test_a_flat_row_major_matrix_is_accepted():
    # OpenCV's FileStorage writes matrices as a flat `data` array, so a calibration
    # converted from one arrives this way and means exactly the same thing.
    document = overhead()
    document["camera_matrix"] = [1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1.0]

    model = CameraModel.from_matrices(**document)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(np.array([[10.0, 0.0]]))


def test_a_column_vector_translation_is_accepted():
    document = overhead()
    document["tvec"] = [[0.0], [0.0], [100.0]]

    model = CameraModel.from_matrices(**document)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(np.array([[10.0, 0.0]]))


# --- reading one from a document ----------------------------------------------

OPENCV_YML = """%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 0., 0., 1000., 0., 0., 0., 1. ]
dist_coeffs: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ 0., 0., 0., 0. ]
rot_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
tvec: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0., 0., 100. ]
"""


def test_a_calibration_is_read_from_the_document_a_calibration_tool_wrote():
    # The same overhead camera as above, in the one format calibrations come in — and
    # it needs no conversion first. `data` is flat and row-major, `tvec` is a column,
    # and dist_coeffs is present and ignored.
    model = CameraModel.from_calibration(OPENCV_YML.encode())

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(
        np.array([[10.0, 0.0]])
    )


def test_a_calibration_can_be_read_from_a_file(tmp_path):
    path = tmp_path / "camera_model.yml"
    path.write_text(OPENCV_YML)

    model = CameraModel.from_calibration(path)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(
        np.array([[10.0, 0.0]])
    )


def test_a_calibration_can_be_read_from_bytes_with_no_file_behind_them():
    # How the detection worker holds one: fetched from object storage, with no path to
    # point at. Writing it to a temp file to be allowed to read it would be absurd.
    model = CameraModel.from_calibration(bytearray(OPENCV_YML.encode()))

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(
        np.array([[10.0, 0.0]])
    )


def test_a_calibration_missing_a_node_is_rejected_by_name():
    # Absent rather than empty, so the error names the field the same way it would for
    # a camera built from matrices directly.
    without_rotation = "\n".join(
        line for line in OPENCV_YML.splitlines() if "rot_matrix" not in line
    )

    with pytest.raises(CalibrationInvalid, match="no rot_matrix"):
        CameraModel.from_calibration(without_rotation.encode())


def test_a_document_that_is_not_a_calibration_is_rejected():
    # OpenCV surfaces a malformed document as a bare SystemError out of its
    # constructor, which is not an exception type any caller should ever see.
    with pytest.raises(CalibrationInvalid, match="not a readable OpenCV calibration"):
        CameraModel.from_calibration(b"not a calibration at all")


def test_a_json_document_is_rejected_as_the_wrong_format():
    # JSON is not read. Calibrations are what calibration tools write, and supporting
    # a second format bought nothing but the code to tell them apart.
    with pytest.raises(CalibrationInvalid):
        CameraModel.from_calibration(b'{"camera_matrix": [[1, 0, 0]]}')


def test_a_calibration_file_that_is_not_there_is_rejected(tmp_path):
    with pytest.raises(CalibrationInvalid, match="cannot read"):
        CameraModel.from_calibration(tmp_path / "absent.yml")
