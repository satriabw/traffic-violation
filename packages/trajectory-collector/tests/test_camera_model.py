import json

import numpy as np
import pytest

from trajectory_collector import CalibrationInvalid, CameraModel


def overhead(focal: float = 1000.0, height: float = 100.0) -> dict:
    """A camera looking straight down from `height` metres.

    Deliberately the simplest calibration that is still a real one: with no rotation
    and the camera on the optical axis, the ground is a plain scaling of the image at
    `height / focal` metres per pixel. Defaults give 0.1 m/px, so every expected value
    in these tests can be worked out by hand.
    """
    return {
        "camera_matrix": [[focal, 0.0, 0.0], [0.0, focal, 0.0], [0.0, 0.0, 1.0]],
        "rot_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tvec": [0.0, 0.0, height],
    }


def test_a_pixel_becomes_a_point_in_metres():
    model = CameraModel.from_calibration(overhead())

    ground = model.project_to_ground(np.array([[100.0, 250.0]]))

    # 0.1 metres per pixel, straight down.
    assert ground == pytest.approx(np.array([[10.0, 25.0]]))


def test_the_whole_frame_projects_in_one_call():
    model = CameraModel.from_calibration(overhead())

    ground = model.project_to_ground(np.array([[0.0, 0.0], [10.0, 20.0], [-30.0, 5.0]]))

    assert ground == pytest.approx(np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]]))


def test_an_empty_frame_projects_to_nothing():
    # Most frames of most footage have something in them, but not all, and an empty
    # array must not become an exception halfway down a video.
    ground = CameraModel.from_calibration(overhead()).project_to_ground(
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
    model = CameraModel.from_calibration(document)
    pixel_from_ground = np.linalg.inv(model.ground_from_pixel)

    truth = np.array([[3.0, 12.0], [-7.5, 40.0]])
    homogeneous = np.column_stack((truth, np.ones(len(truth)))) @ pixel_from_ground.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]

    assert model.project_to_ground(pixels) == pytest.approx(truth)


def test_further_away_is_more_metres_per_pixel():
    # The property that makes projecting worth doing at all: two objects the same
    # distance apart on screen are not the same distance apart on the ground.
    model = CameraModel.from_calibration(_tilted())

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
    ground = CameraModel.from_calibration(_tilted()).project_to_ground(
        np.array([[0.0, 0.0]])
    )

    assert not np.isfinite(ground).all()


def test_one_pixel_on_the_horizon_does_not_spoil_the_rest_of_the_frame():
    # The whole frame projects in one multiply, so a single unusable box must not take
    # its neighbours with it.
    ground = CameraModel.from_calibration(_tilted()).project_to_ground(
        np.array([[0.0, 500.0], [0.0, 0.0], [0.0, 400.0]])
    )

    assert np.isfinite(ground[[0, 2]]).all()
    assert not np.isfinite(ground[1]).all()


# --- documents that cannot be used --------------------------------------------


def test_a_missing_field_is_rejected_at_construction():
    # Not on some frame in the middle of a video. A calibration is checked once and
    # used tens of thousands of times.
    document = overhead()
    del document["rot_matrix"]

    with pytest.raises(CalibrationInvalid, match="no rot_matrix"):
        CameraModel.from_calibration(document)


def test_a_field_of_the_wrong_shape_is_rejected():
    document = overhead()
    document["camera_matrix"] = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(CalibrationInvalid, match="shape"):
        CameraModel.from_calibration(document)


def test_a_field_that_is_not_numbers_is_rejected():
    document = overhead()
    document["tvec"] = ["over", "there", "somewhere"]

    with pytest.raises(CalibrationInvalid, match="not numeric"):
        CameraModel.from_calibration(document)


def test_a_camera_with_no_invertible_ground_plane_is_rejected():
    # Physically: the ground has no image to invert. Numerically: a singular matrix,
    # which would otherwise surface as a LinAlgError from the first frame of a video.
    document = overhead()
    document["tvec"] = [0.0, 0.0, 0.0]

    with pytest.raises(CalibrationInvalid, match="ground plane"):
        CameraModel.from_calibration(document)


def test_a_flat_row_major_matrix_is_accepted():
    # OpenCV's FileStorage writes matrices as a flat `data` array, so a calibration
    # converted from one arrives this way and means exactly the same thing.
    document = overhead()
    document["camera_matrix"] = [1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1.0]

    model = CameraModel.from_calibration(document)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(np.array([[10.0, 0.0]]))


def test_a_column_vector_translation_is_accepted():
    document = overhead()
    document["tvec"] = [[0.0], [0.0], [100.0]]

    model = CameraModel.from_calibration(document)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(np.array([[10.0, 0.0]]))


# --- reading one from disk ----------------------------------------------------


def test_a_calibration_can_be_read_from_a_json_file(tmp_path):
    path = tmp_path / "camera_model.json"
    path.write_text(json.dumps(overhead()))

    model = CameraModel.from_calibration(path)

    assert model.project_to_ground(np.array([[100.0, 0.0]])) == pytest.approx(np.array([[10.0, 0.0]]))


def test_an_opencv_yml_calibration_says_what_to_do_about_it(tmp_path):
    # Reading one needs OpenCV, which this package deliberately does not depend on.
    # The error has to say so, because the file is not wrong — only unsupported here.
    path = tmp_path / "camera_model.yml"
    path.write_text("%YAML:1.0\n")

    with pytest.raises(CalibrationInvalid, match="convert it to JSON"):
        CameraModel.from_calibration(path)


def test_a_calibration_file_that_is_not_there_is_rejected(tmp_path):
    with pytest.raises(CalibrationInvalid, match="cannot read"):
        CameraModel.from_calibration(tmp_path / "absent.json")


def test_a_calibration_file_that_is_not_json_is_rejected(tmp_path):
    path = tmp_path / "camera_model.json"
    path.write_text("{not json at all")

    with pytest.raises(CalibrationInvalid, match="not valid JSON"):
        CameraModel.from_calibration(path)
