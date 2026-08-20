"""The object detector: one frame in, detections out.

RT-DETR through onnxruntime, and deliberately nothing else. The research pipeline this
is ported from reaches inference via torch, torchvision, transformers and PIL — all
four imported at module scope, even on the ONNX path, purely to resize a frame and
scale it to [0,1]. That is a multi-gigabyte dependency doing what a cv2 resize and a
divide do below, so none of it comes across.

One behavioural difference is worth knowing about: torchvision's Resize antialiases and
`cv2.resize` with INTER_LINEAR does not, so boxes can land a pixel or two from where
the published numbers put them. Worth remembering when comparing against the paper.

`sv.Detections` is the return type rather than something of our own because it is what
`sv.ByteTrack` consumes, and what the rule engine will expect when it is ported.
"""

import logging
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

from shared import config

logger = logging.getLogger(__name__)

# The export takes a fixed square tensor; the model's own `orig_target_sizes` input is
# what undoes the distortion, so there is no letterboxing to do here.
INPUT_SIZE = 640

# COCO's 80-class contiguous layout, which is what this export actually emits. A
# constructor argument rather than a fixed module constant because a model registry
# means several models with different label spaces coexisting — the map has to be able
# to travel with the model.
#
# Traffic lights are 9 here, where the research pipeline's map says 10 (the id from
# COCO's 91-class layout). Verified against the real export: 30 frames of its demo
# footage yield class ids 2 and 9 only, and nothing above 9. The discrepancy is dormant
# in that pipeline because it reads traffic light state from annotated polygons rather
# than from detections, so its `TRAFFIC_LIGHT = 10` is never matched against a real
# detection. It would stop being dormant the moment anything here detects one.
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
    9: "traffic_light",
}


class ModelUnavailable(Exception):
    """No usable model file at the reference we were given."""


class DetectionModel(Protocol):
    """What the worker needs from a detector, and all it needs.

    Narrow on purpose: a fake satisfying this one method lets the whole pipeline below
    it be exercised without onnxruntime, a GPU, or a weights file.
    """

    def predict(self, frame: np.ndarray) -> sv.Detections: ...


def resolve_model_path(ref: str) -> Path:
    """The local file to load the model from.

    A local path today. Pulling from R2 belongs here too — download to a cache
    directory keyed by the object key, return the cached path — and putting the seam in
    now means adding it later touches this function and nothing that calls it.
    """
    path = Path(ref).expanduser()
    if not path.is_file():
        raise ModelUnavailable(f"no model file at {path}")
    return path


class RTDetrOnnx:
    def __init__(
        self,
        model_path: str,
        threshold: float | None = None,
        providers: list[str] | None = None,
        class_names: dict[int, str] | None = None,
        session_factory: Callable[..., ort.InferenceSession] = ort.InferenceSession,
    ):
        """Building this is expensive — the session parses the graph and allocates for
        the execution provider — so it is built once per process and injected into the
        handler, never per job.

        `session_factory` is a parameter for the same reason `capture` is one in
        reader.py: so the tests drive predict() without onnxruntime or a real model.
        """
        self._path = resolve_model_path(model_path)
        self._threshold = config.DETECTION_THRESHOLD if threshold is None else threshold
        self._class_names = COCO_CLASSES if class_names is None else class_names
        self._session = session_factory(
            str(self._path),
            providers=providers or config.DETECTION_MODEL_PROVIDERS,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # BGR is what OpenCV decodes to and RGB is what the model was trained on.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # HWC uint8 -> NCHW float32 in [0,1].
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    def predict(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]

        labels, boxes, scores = self._session.run(
            None,
            {
                "images": self._preprocess(frame),
                # The frame's real size, not INPUT_SIZE. The model scales its boxes by
                # this, so it is what brings them back into the coordinate space the
                # ROI polygons are drawn in.
                "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
            },
        )

        # One mask across all three outputs, so a box can never end up paired with
        # another detection's class.
        keep = scores >= self._threshold
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        if len(boxes) == 0:
            # The common case, frame to frame. It has to be an empty Detections the
            # tracker can still be updated with — skipping the update would let the
            # tracker's idea of "now" drift from the video's.
            return sv.Detections.empty()

        class_id = labels.astype(np.int16)
        return sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=class_id,
            # An id outside the map still yields a detection, named by its number. An
            # unrecognised class is not a reason to drop a box; deciding which types
            # matter is the rule engine's job, not the detector's.
            data={
                "class_name": np.array(
                    [self._class_names.get(int(c), str(int(c))) for c in class_id]
                )
            },
        )


def from_config() -> RTDetrOnnx:
    """The detector this deployment is configured for."""
    if not config.DETECTION_MODEL_PATH:
        raise ModelUnavailable("DETECTION_MODEL_PATH is not set")
    logger.info("loading detection model from %s", config.DETECTION_MODEL_PATH)
    return RTDetrOnnx(config.DETECTION_MODEL_PATH)
