"""SCRFD detector adapter, registered as an opt-in entrant.

SCRFD is anchor-free with two anchors per feature-map cell at strides 8, 16 and 32. Box
and landmark heads predict distances from the anchor centre, expressed in stride units,
so decoding is a subtraction and addition around that centre rather than YuNet's
exponential extent.

The weights are released for non-commercial research use only (see decision D008), which
is why this adapter is available but never the default.
"""

from __future__ import annotations

import numpy as np

from portraitkit.detection.base import FaceDetector, RawDetections
from portraitkit.errors import ModelError
from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.types import ImageSize

__all__ = ["ANCHORS_PER_CELL", "STRIDES", "ScrfdDetector", "decode_scrfd"]

STRIDES: tuple[int, ...] = (8, 16, 32)
"""Feature-map strides, in the order the artifact emits them."""

ANCHORS_PER_CELL = 2
"""SCRFD places two anchors at every feature-map location."""

INPUT_SIZE = ImageSize(width=640, height=640)

SCRFD_CONTRACT = PreprocessContract(
    input_name="input.1",
    input_size=INPUT_SIZE,
    # InsightFace swaps to RGB and maps 0-255 onto roughly [-1, 1].
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(127.5, 127.5, 127.5),
    scale=1.0 / 128.0,
    resize_mode=ResizeMode.LETTERBOX_TOP_LEFT,
)


def _anchor_centres(width: int, height: int, stride: int, anchors: int) -> np.ndarray:
    """Return ``(width * height * anchors, 2)`` anchor centres in pixel coordinates."""
    rows, columns = np.mgrid[:height, :width]
    centres = np.stack([columns, rows], axis=-1).astype(np.float32) * stride
    centres = centres.reshape(-1, 2)
    if anchors > 1:
        centres = np.repeat(centres, anchors, axis=0)
    return centres


def decode_scrfd(
    outputs: list[np.ndarray],
    *,
    input_size: ImageSize = INPUT_SIZE,
    strides: tuple[int, ...] = STRIDES,
    anchors_per_cell: int = ANCHORS_PER_CELL,
) -> RawDetections:
    """Decode SCRFD output tensors into candidate boxes, scores, and landmarks.

    Args:
        outputs: Nine tensors ordered ``score``, ``bbox``, ``kps``, each group ascending
            by stride.
        input_size: Spatial size the tensors were produced at.
        strides: Feature-map strides matching the tensor groups.
        anchors_per_cell: Anchors placed at each feature-map location.

    Returns:
        Candidates in model-input pixel coordinates.
    """
    expected = 3 * len(strides)
    if len(outputs) != expected:
        msg = f"SCRFD expects {expected} output tensors, got {len(outputs)}"
        raise ModelError(msg)

    count = len(strides)
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []

    for index, stride in enumerate(strides):
        confidence = np.asarray(outputs[index], dtype=np.float32).reshape(-1)
        # Distance heads are predicted in stride units.
        box_distances = np.asarray(outputs[index + count], dtype=np.float32).reshape(-1, 4) * stride
        point_distances = (
            np.asarray(outputs[index + 2 * count], dtype=np.float32).reshape(-1, 10) * stride
        )

        grid_width = input_size.width // stride
        grid_height = input_size.height // stride
        expected_rows = grid_width * grid_height * anchors_per_cell
        if confidence.size != expected_rows:
            msg = (
                f"SCRFD stride {stride} produced {confidence.size} rows, expected "
                f"{expected_rows} for a {input_size.width}x{input_size.height} input "
                f"with {anchors_per_cell} anchors per cell"
            )
            raise ModelError(msg)

        centres = _anchor_centres(grid_width, grid_height, stride, anchors_per_cell)
        centre_x = centres[:, 0]
        centre_y = centres[:, 1]

        boxes.append(
            np.stack(
                [
                    centre_x - box_distances[:, 0],
                    centre_y - box_distances[:, 1],
                    centre_x + box_distances[:, 2],
                    centre_y + box_distances[:, 3],
                ],
                axis=-1,
            )
        )
        scores.append(confidence)

        points = np.empty((point_distances.shape[0], 5, 2), dtype=np.float32)
        points[:, :, 0] = centre_x[:, np.newaxis] + point_distances[:, 0::2]
        points[:, :, 1] = centre_y[:, np.newaxis] + point_distances[:, 1::2]
        landmarks.append(points)

    return RawDetections(
        boxes=np.concatenate(boxes, axis=0),
        scores=np.concatenate(scores, axis=0),
        landmarks=np.concatenate(landmarks, axis=0),
    )


class ScrfdDetector(FaceDetector):
    """Adapter for the SCRFD-10G artifact."""

    @property
    def name(self) -> str:
        return "scrfd-10g-bnkps"

    @property
    def contract(self) -> PreprocessContract:
        return SCRFD_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> RawDetections:
        return decode_scrfd(
            outputs,
            input_size=self.contract.input_size,
            strides=STRIDES,
            anchors_per_cell=ANCHORS_PER_CELL,
        )
