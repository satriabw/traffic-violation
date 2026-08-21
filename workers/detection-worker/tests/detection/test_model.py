import os

import numpy as np
import pytest
import supervision as sv

from detection_worker.detection.model import (
    COCO_CLASSES,
    ModelUnavailable,
    RTDetrOnnx,
    resolve_model_path,
)


class FakeSession:
    """An ORT session with the one method RTDetrOnnx calls.

    It records the input_feed it was handed, which is the only way to check the
    preprocessing from outside: everything the model does to a frame before inference
    ends up in that dict, and nothing else observes it.
    """

    def __init__(self, labels=None, boxes=None, scores=None):
        self.input_feed = None
        self.output_names = "unset"
        self._outputs = (
            np.array([[0]], dtype=np.int64) if labels is None else labels,
            np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32) if boxes is None else boxes,
            np.array([[0.9]], dtype=np.float32) if scores is None else scores,
        )

    def run(self, output_names, input_feed):
        self.output_names = output_names
        self.input_feed = input_feed
        return self._outputs


def _model(path, session, **kwargs):
    return RTDetrOnnx(path, session_factory=lambda *a, **k: session, **kwargs)


def _frame(width=100, height=50, bgr=(10, 20, 30)):
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = bgr
    return frame


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "rt-detr.onnx"
    path.write_bytes(b"not a real model, the session is faked")
    return path


# --- resolving the model ------------------------------------------------------


def test_a_local_path_resolves_to_itself(model_file):
    assert resolve_model_path(str(model_file)) == model_file


def test_a_missing_local_path_raises(tmp_path):
    # Loudly at startup, rather than as an opaque ORT error on the first frame of the
    # first job — by which point a worker has already claimed work it cannot do.
    with pytest.raises(ModelUnavailable):
        resolve_model_path(str(tmp_path / "absent.onnx"))


def test_a_directory_is_not_a_model(tmp_path):
    with pytest.raises(ModelUnavailable):
        resolve_model_path(str(tmp_path))


def test_constructing_the_model_resolves_the_path_eagerly(tmp_path):
    with pytest.raises(ModelUnavailable):
        _model(str(tmp_path / "absent.onnx"), FakeSession())


# --- preprocessing ------------------------------------------------------------


def test_the_image_tensor_has_the_shape_and_dtype_the_model_expects(model_file):
    session = FakeSession()

    _model(str(model_file), session).predict(_frame())

    images = session.input_feed["images"]
    assert images.shape == (1, 3, 640, 640)
    assert images.dtype == np.float32


def test_pixels_are_scaled_into_zero_to_one(model_file):
    session = FakeSession()

    _model(str(model_file), session).predict(_frame(bgr=(0, 128, 255)))

    images = session.input_feed["images"]
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_channels_are_reordered_from_opencv_bgr_to_rgb(model_file):
    # OpenCV hands us BGR and the model was trained on RGB. Getting this backwards
    # costs no error and a lot of accuracy, so it is worth pinning with real values.
    session = FakeSession()

    _model(str(model_file), session).predict(_frame(bgr=(10, 20, 30)))

    images = session.input_feed["images"]
    assert images[0, 0].mean() == pytest.approx(30 / 255, abs=1e-3)  # R
    assert images[0, 1].mean() == pytest.approx(20 / 255, abs=1e-3)  # G
    assert images[0, 2].mean() == pytest.approx(10 / 255, abs=1e-3)  # B


def test_orig_target_sizes_carries_the_frames_own_width_and_height(model_file):
    # Not the 640x640 the tensor was resized to. This input is what the model uses to
    # scale its boxes back, so getting it wrong returns boxes in the wrong coordinate
    # space — silently, and every downstream ROI test would then be wrong too.
    session = FakeSession()

    _model(str(model_file), session).predict(_frame(width=1920, height=1080))

    assert session.input_feed["orig_target_sizes"].tolist() == [[1920, 1080]]
    assert session.input_feed["orig_target_sizes"].dtype == np.int64


# --- postprocessing -----------------------------------------------------------


def _outputs(*triples):
    """Build (labels, boxes, scores) in the batch-of-one shape the model returns."""
    labels = np.array([[label for label, _, _ in triples]], dtype=np.int64)
    boxes = np.array([[box for _, box, _ in triples]], dtype=np.float32)
    scores = np.array([[score for _, _, score in triples]], dtype=np.float32)
    return labels, boxes, scores


def test_detections_above_the_threshold_survive(model_file):
    session = FakeSession(*_outputs(
        (2, [0, 0, 10, 10], 0.9),
        (0, [5, 5, 15, 15], 0.8),
    ))

    detections = _model(str(model_file), session, threshold=0.5).predict(_frame())

    assert len(detections) == 2
    assert detections.xyxy.tolist() == [[0, 0, 10, 10], [5, 5, 15, 15]]
    assert detections.class_id.tolist() == [2, 0]


def test_detections_below_the_threshold_are_dropped_and_the_rest_stay_aligned(model_file):
    # The three arrays are filtered by one mask. If they ever drift apart, a box gets
    # the wrong class, which is the kind of bug that survives a long time.
    session = FakeSession(*_outputs(
        (2, [0, 0, 10, 10], 0.9),
        (7, [1, 1, 11, 11], 0.2),
        (0, [5, 5, 15, 15], 0.8),
    ))

    detections = _model(str(model_file), session, threshold=0.5).predict(_frame())

    assert detections.xyxy.tolist() == [[0, 0, 10, 10], [5, 5, 15, 15]]
    assert detections.class_id.tolist() == [2, 0]
    assert detections.confidence.tolist() == pytest.approx([0.9, 0.8])


def test_a_frame_with_nothing_above_the_threshold_yields_no_detections(model_file):
    # Empty is normal — most frames of most videos. It must be an empty Detections
    # the tracker can still be updated with, not None and not an exception.
    session = FakeSession(*_outputs((2, [0, 0, 10, 10], 0.1)))

    detections = _model(str(model_file), session, threshold=0.5).predict(_frame())

    assert len(detections) == 0
    assert isinstance(detections, sv.Detections)


def test_a_model_that_returns_nothing_at_all_yields_no_detections(model_file):
    session = FakeSession(
        np.zeros((1, 0), dtype=np.int64),
        np.zeros((1, 0, 4), dtype=np.float32),
        np.zeros((1, 0), dtype=np.float32),
    )

    assert len(_model(str(model_file), session).predict(_frame())) == 0


def test_class_ids_are_int16(model_file):
    # What ByteTrack's internals expect to index with.
    session = FakeSession(*_outputs((2, [0, 0, 10, 10], 0.9)))

    detections = _model(str(model_file), session).predict(_frame())

    assert detections.class_id.dtype == np.int16


def test_known_class_ids_are_named_for_the_log(model_file):
    session = FakeSession(*_outputs(
        (2, [0, 0, 10, 10], 0.9),
        (999, [5, 5, 15, 15], 0.9),
    ))

    detections = _model(str(model_file), session).predict(_frame())

    # An id the map does not cover still yields a detection — an unnamed class is not
    # a reason to drop a box, and filtering by type is the rule engine's job.
    assert list(detections.data["class_name"]) == [COCO_CLASSES[2], "999"]


def test_the_class_map_can_be_replaced(model_file):
    # A registry means several models with different label spaces, so the map has to
    # be able to travel with the model rather than being a module-level constant.
    session = FakeSession(*_outputs((4, [0, 0, 10, 10], 0.9)))

    detections = _model(str(model_file), session, class_names={4: "tram"}).predict(_frame())

    assert list(detections.data["class_name"]) == ["tram"]


# --- against a real onnxruntime session ---------------------------------------

_REAL_MODEL = os.environ.get("DETECTION_MODEL_PATH", "")


@pytest.mark.skipif(not _REAL_MODEL, reason="set DETECTION_MODEL_PATH to a real .onnx")
def test_runs_against_a_real_model():
    """The only test that checks our input names and dtypes against a real model.

    Everything above is written against a fake we also wrote, so it cannot catch an
    input named `images` that the model actually calls something else.
    """
    model = RTDetrOnnx(_REAL_MODEL)

    detections = model.predict(_frame(width=640, height=480))

    assert isinstance(detections, sv.Detections)
    if len(detections):
        assert detections.xyxy.shape[1] == 4
