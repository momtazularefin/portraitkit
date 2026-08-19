"""YuNet detector adapter, PortraitKit's default.

YuNet is an anchor-free detector with one prediction per feature-map cell at strides 8,
16 and 32. Each cell emits a classification score, an objectness score, a box offset, and
five landmark offsets. Confidence is the geometric mean of the two scores, which is how
the reference implementation combines them.

Box offsets are centre-relative in cell units with log-scale extents; landmark offsets
are cell-relative in cell units. All of it is decoded here in one place so the arithmetic
is inspectable and testable without a model file.
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

__all__ = ["STRIDES", "YuNetDetector", "decode_yunet"]

STRIDES: tuple[int, ...] = (8, 16, 32)
"""Feature-map strides, in the order the artifact emits them."""

INPUT_SIZE = ImageSize(width=640, height=640)

YUNET_CONTRACT = PreprocessContract(
    input_name="input",
    input_size=INPUT_SIZE,
    # The reference wrapper feeds OpenCV's native BGR with neither mean subtraction nor
    # scaling, so raw 0-255 values reach the network.
    color_order=ColorOrder.BGR,
    layout=TensorLayout.NCHW,
    mean=(0.0, 0.0, 0.0),
    scale=1.0,
    resize_mode=ResizeMode.LETTERBOX_TOP_LEFT,
)


def _cell_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell column and row indices in row-major order."""
    rows, columns = np.divmod(np.arange(width * height, dtype=np.float32), width)
    return columns, rows


def decode_yunet(
    outputs: list[np.ndarray],
    *,
    input_size: ImageSize = INPUT_SIZE,
    strides: tuple[int, ...] = STRIDES,
) -> RawDetections:
    """Decode YuNet output tensors into candidate boxes, scores, and landmarks.

    Args:
        outputs: The twelve output tensors, ordered ``cls``, ``obj``, ``bbox``, ``kps``,
            each group ascending by stride.
        input_size: Spatial size the tensors were produced at.
        strides: Feature-map strides matching the tensor groups.

    Returns:
        Candidates in model-input pixel coordinates.
    """
    expected = 4 * len(strides)
    if len(outputs) != expected:
        msg = f"YuNet expects {expected} output tensors, got {len(outputs)}"
        raise ModelError(msg)

    count = len(strides)
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []

    for index, stride in enumerate(strides):
        classification = np.asarray(outputs[index], dtype=np.float32).reshape(-1)
        objectness = np.asarray(outputs[index + count], dtype=np.float32).reshape(-1)
        box_offsets = np.asarray(outputs[index + 2 * count], dtype=np.float32).reshape(-1, 4)
        point_offsets = np.asarray(outputs[index + 3 * count], dtype=np.float32).reshape(-1, 10)

        grid_width = input_size.width // stride
        grid_height = input_size.height // stride
        cells = grid_width * grid_height
        if classification.size != cells:
            msg = (
                f"YuNet stride {stride} produced {classification.size} cells, "
                f"expected {cells} for a {input_size.width}x{input_size.height} input"
            )
            raise ModelError(msg)

        columns, rows = _cell_grid(grid_width, grid_height)

        # Confidence combines the two heads; clipping guards against tiny negative
        # products from quantization noise before the square root.
        confidence = np.sqrt(np.clip(classification * objectness, 0.0, 1.0))

        centre_x = (columns + box_offsets[:, 0]) * stride
        centre_y = (rows + box_offsets[:, 1]) * stride
        width = np.exp(box_offsets[:, 2]) * stride
        height = np.exp(box_offsets[:, 3]) * stride

        boxes.append(
            np.stack(
                [
                    centre_x - width / 2.0,
                    centre_y - height / 2.0,
                    centre_x + width / 2.0,
                    centre_y + height / 2.0,
                ],
                axis=-1,
            )
        )
        scores.append(confidence)

        points = np.empty((point_offsets.shape[0], 5, 2), dtype=np.float32)
        points[:, :, 0] = (columns[:, np.newaxis] + point_offsets[:, 0::2]) * stride
        points[:, :, 1] = (rows[:, np.newaxis] + point_offsets[:, 1::2]) * stride
        landmarks.append(points)

    return RawDetections(
        boxes=np.concatenate(boxes, axis=0),
        scores=np.concatenate(scores, axis=0),
        landmarks=np.concatenate(landmarks, axis=0),
    )


class YuNetDetector(FaceDetector):
    """Adapter for the OpenCV Zoo YuNet artifact."""

    @property
    def name(self) -> str:
        return "yunet-2023mar"

    @property
    def contract(self) -> PreprocessContract:
        return YUNET_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> RawDetections:
        return decode_yunet(outputs, input_size=self.contract.input_size, strides=STRIDES)
