"""Non-maximum suppression."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.detection.nms import non_max_suppression


def boxes_of(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_empty_input_returns_empty_selection() -> None:
    keep = non_max_suppression(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        iou_threshold=0.5,
    )

    assert keep.shape == (0,)


def test_single_box_survives() -> None:
    keep = non_max_suppression(
        boxes_of((0.0, 0.0, 10.0, 10.0)), np.asarray([0.9], dtype=np.float32), iou_threshold=0.5
    )

    assert keep.tolist() == [0]


def test_duplicate_is_suppressed_and_the_better_score_wins() -> None:
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (0.5, 0.5, 10.5, 10.5))
    scores = np.asarray([0.7, 0.95], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert keep.tolist() == [1]


def test_disjoint_boxes_both_survive() -> None:
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0))
    scores = np.asarray([0.8, 0.6], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert sorted(keep.tolist()) == [0, 1]


def test_results_are_ordered_by_descending_score() -> None:
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0), (100.0, 100.0, 110.0, 110.0))
    scores = np.asarray([0.3, 0.9, 0.6], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert keep.tolist() == [1, 2, 0]


def test_threshold_decides_borderline_overlap() -> None:
    # Half-overlapping boxes: intersection 50, union 150, IoU = 1/3.
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0))
    scores = np.asarray([0.9, 0.8], dtype=np.float32)

    assert non_max_suppression(boxes, scores, iou_threshold=0.4).tolist() == [0, 1]
    assert non_max_suppression(boxes, scores, iou_threshold=0.2).tolist() == [0]


def test_top_k_bounds_the_candidates_considered() -> None:
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0), (100.0, 100.0, 110.0, 110.0))
    scores = np.asarray([0.3, 0.9, 0.6], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5, top_k=2)

    assert keep.tolist() == [1, 2]


def test_zero_area_boxes_are_treated_as_disjoint() -> None:
    """Degenerate candidates must not divide by a zero union."""
    boxes = boxes_of((5.0, 5.0, 5.0, 5.0), (5.0, 5.0, 5.0, 5.0))
    scores = np.asarray([0.9, 0.8], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert sorted(keep.tolist()) == [0, 1]


def test_identical_boxes_collapse_to_one() -> None:
    boxes = boxes_of((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0))
    scores = np.asarray([0.5, 0.9, 0.7], dtype=np.float32)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert keep.tolist() == [1]


def test_malformed_box_array_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        non_max_suppression(
            np.zeros((3, 5), dtype=np.float32), np.zeros((3,), dtype=np.float32), iou_threshold=0.5
        )


def test_mismatched_score_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 2 scores"):
        non_max_suppression(
            boxes_of((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)),
            np.zeros((3,), dtype=np.float32),
            iou_threshold=0.5,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_out_of_range_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"iou_threshold must be in \[0, 1\]"):
        non_max_suppression(
            boxes_of((0.0, 0.0, 1.0, 1.0)),
            np.asarray([0.5], dtype=np.float32),
            iou_threshold=threshold,
        )
