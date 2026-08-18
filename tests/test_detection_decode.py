"""Decoder arithmetic for both detector adapters.

These tests assert the exact box and landmark values the published decoding formulas
imply, using hand-built output tensors and a small input size. Getting anchor geometry
subtly wrong produces detections that still look plausible on a photograph, so the
arithmetic is pinned to arithmetic rather than to eyeballing a picture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from portraitkit.detection.scrfd import decode_scrfd
from portraitkit.detection.yunet import decode_yunet
from portraitkit.errors import ModelError
from portraitkit.types import ImageSize

SIZE = ImageSize(width=64, height=64)
STRIDES = (8, 16, 32)


def yunet_outputs() -> list[np.ndarray]:
    """Zeroed YuNet tensors for a 64x64 input: cls, obj, bbox, kps per stride."""
    tensors: list[np.ndarray] = []
    for group_width in (1, 1, 4, 10):
        for stride in STRIDES:
            cells = (SIZE.width // stride) * (SIZE.height // stride)
            tensors.append(np.zeros((1, cells, group_width), dtype=np.float32))
    return tensors


def scrfd_outputs() -> list[np.ndarray]:
    """Zeroed SCRFD tensors for a 64x64 input: score, bbox, kps per stride."""
    tensors: list[np.ndarray] = []
    for group_width in (1, 4, 10):
        for stride in STRIDES:
            rows = (SIZE.width // stride) * (SIZE.height // stride) * 2
            tensors.append(np.zeros((rows, group_width), dtype=np.float32))
    return tensors


def test_yunet_decodes_a_known_cell_exactly() -> None:
    outputs = yunet_outputs()
    cell = 10  # column 2, row 1 on the stride-8 grid, which is 8 cells wide

    outputs[0][0, cell, 0] = 1.0  # cls_8
    outputs[3][0, cell, 0] = 1.0  # obj_8
    outputs[6][0, cell] = [0.5, 0.5, math.log(2.0), math.log(3.0)]  # bbox_8
    outputs[9][0, cell] = [0.25, 0.75] * 5  # kps_8

    raw = decode_yunet(outputs, input_size=SIZE, strides=STRIDES)

    assert raw.scores[cell] == pytest.approx(1.0)
    # centre (2 + 0.5) * 8 = 20, (1 + 0.5) * 8 = 12; extent exp(log 2) * 8 = 16 by 24.
    assert raw.boxes[cell] == pytest.approx([12.0, 0.0, 28.0, 24.0])
    assert raw.landmarks is not None
    assert raw.landmarks[cell, 0] == pytest.approx([18.0, 14.0])
    assert raw.landmarks[cell, 4] == pytest.approx([18.0, 14.0])


def test_yunet_score_is_the_geometric_mean_of_both_heads() -> None:
    outputs = yunet_outputs()
    outputs[0][0, 0, 0] = 0.25
    outputs[3][0, 0, 0] = 0.81

    raw = decode_yunet(outputs, input_size=SIZE, strides=STRIDES)

    assert raw.scores[0] == pytest.approx(math.sqrt(0.25 * 0.81))


def test_yunet_clamps_negative_score_products() -> None:
    """Quantization noise can push a head slightly negative; the square root must not."""
    outputs = yunet_outputs()
    outputs[0][0, 0, 0] = -0.01
    outputs[3][0, 0, 0] = 0.5

    raw = decode_yunet(outputs, input_size=SIZE, strides=STRIDES)

    assert raw.scores[0] == pytest.approx(0.0)


def test_yunet_covers_every_cell_across_all_strides() -> None:
    raw = decode_yunet(yunet_outputs(), input_size=SIZE, strides=STRIDES)

    expected = sum((SIZE.width // stride) * (SIZE.height // stride) for stride in STRIDES)
    assert raw.scores.shape == (expected,)
    assert raw.boxes.shape == (expected, 4)
    assert raw.landmarks is not None
    assert raw.landmarks.shape == (expected, 5, 2)


def test_yunet_rejects_a_wrong_tensor_count() -> None:
    with pytest.raises(ModelError, match="expects 12 output tensors"):
        decode_yunet(yunet_outputs()[:-1], input_size=SIZE, strides=STRIDES)


def test_yunet_rejects_a_grid_that_does_not_match_the_input_size() -> None:
    with pytest.raises(ModelError, match="expected 6400"):
        decode_yunet(yunet_outputs(), input_size=ImageSize(width=640, height=640))


def test_scrfd_decodes_a_known_anchor_exactly() -> None:
    outputs = scrfd_outputs()
    row = 21  # cell 10 (column 2, row 1) on the stride-8 grid, second anchor

    outputs[0][row, 0] = 0.9  # score_8
    outputs[3][row] = [1.0, 0.5, 2.0, 1.5]  # bbox_8, distances in stride units
    outputs[6][row] = [0.5, -0.25] * 5  # kps_8

    raw = decode_scrfd(outputs, input_size=SIZE, strides=STRIDES)

    assert raw.scores[row] == pytest.approx(0.9)
    # anchor centre (2 * 8, 1 * 8) = (16, 8); distances scale by the stride.
    assert raw.boxes[row] == pytest.approx([8.0, 4.0, 32.0, 20.0])
    assert raw.landmarks is not None
    assert raw.landmarks[row, 0] == pytest.approx([20.0, 6.0])


def test_scrfd_places_two_anchors_at_every_location() -> None:
    outputs = scrfd_outputs()
    outputs[3][0] = [1.0, 1.0, 1.0, 1.0]
    outputs[3][1] = [1.0, 1.0, 1.0, 1.0]

    raw = decode_scrfd(outputs, input_size=SIZE, strides=STRIDES)

    # Both anchors of cell 0 share a centre, so they decode to the same box.
    assert raw.boxes[0] == pytest.approx(raw.boxes[1])
    expected = sum((SIZE.width // stride) * (SIZE.height // stride) * 2 for stride in STRIDES)
    assert raw.scores.shape == (expected,)


def test_scrfd_rejects_a_wrong_tensor_count() -> None:
    with pytest.raises(ModelError, match="expects 9 output tensors"):
        decode_scrfd(scrfd_outputs()[:-1], input_size=SIZE, strides=STRIDES)


def test_scrfd_rejects_a_mismatched_anchor_count() -> None:
    with pytest.raises(ModelError, match="anchors per cell"):
        decode_scrfd(scrfd_outputs(), input_size=SIZE, strides=STRIDES, anchors_per_cell=1)


@pytest.mark.parametrize("stride_index", [0, 1, 2])
def test_adjacent_scrfd_cells_are_one_stride_apart(stride_index: int) -> None:
    """A stride-32 grid must step 32 pixels per cell, not 8."""
    stride = STRIDES[stride_index]
    outputs = scrfd_outputs()
    block_start = sum((SIZE.width // s) * (SIZE.height // s) * 2 for s in STRIDES[:stride_index])

    raw = decode_scrfd(outputs, input_size=SIZE, strides=STRIDES)

    # With all distances zero, every box collapses onto its anchor centre.
    first_cell_x = raw.boxes[block_start, 0]
    second_cell_x = raw.boxes[block_start + 2, 0]
    assert second_cell_x - first_cell_x == pytest.approx(float(stride))


@pytest.mark.parametrize("stride_index", [0, 1, 2])
def test_adjacent_yunet_cells_are_one_stride_apart(stride_index: int) -> None:
    stride = STRIDES[stride_index]
    outputs = yunet_outputs()
    block_start = sum((SIZE.width // s) * (SIZE.height // s) for s in STRIDES[:stride_index])

    raw = decode_yunet(outputs, input_size=SIZE, strides=STRIDES)

    first_centre = raw.boxes[block_start, 0] + raw.boxes[block_start, 2]
    second_centre = raw.boxes[block_start + 1, 0] + raw.boxes[block_start + 1, 2]
    assert (second_centre - first_centre) / 2.0 == pytest.approx(float(stride))
