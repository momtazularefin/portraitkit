"""Detection counts and landmark error."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.eval.metrics import DetectionCounts, normalized_landmark_error
from portraitkit.types import FaceLandmarks5


def landmarks(*points: tuple[float, float]) -> FaceLandmarks5:
    return FaceLandmarks5.from_array(np.asarray(points, dtype=np.float32))


REFERENCE = landmarks((40, 50), (60, 50), (50, 62), (42, 74), (58, 74))


def test_counts_start_at_zero() -> None:
    counts = DetectionCounts()

    assert counts.predicted == 0
    assert counts.actual == 0
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0


def test_precision_recall_and_f1_from_hand_computed_counts() -> None:
    # 6 correct, 2 spurious, 3 missed: precision 6/8, recall 6/9.
    counts = DetectionCounts(true_positives=6, false_positives=2, false_negatives=3)

    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(2 / 3)
    assert counts.f1 == pytest.approx(2 * 0.75 * (2 / 3) / (0.75 + 2 / 3))


def test_perfect_detection_scores_one() -> None:
    counts = DetectionCounts(true_positives=5)

    assert counts.precision == pytest.approx(1.0)
    assert counts.recall == pytest.approx(1.0)
    assert counts.f1 == pytest.approx(1.0)


def test_predicting_nothing_gives_zero_recall_and_zero_precision() -> None:
    counts = DetectionCounts(false_negatives=4)

    assert counts.predicted == 0
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0


def test_counts_aggregate_by_addition() -> None:
    first = DetectionCounts(true_positives=2, false_positives=1, false_negatives=0)
    second = DetectionCounts(true_positives=3, false_positives=0, false_negatives=2)

    total = first + second

    assert total == DetectionCounts(true_positives=5, false_positives=1, false_negatives=2)


def test_aggregating_counts_is_not_averaging_ratios() -> None:
    """Summing counts before dividing keeps a one-face image from outvoting a busy one."""
    sparse = DetectionCounts(true_positives=1, false_positives=0, false_negatives=0)
    dense = DetectionCounts(true_positives=1, false_positives=9, false_negatives=0)

    total = sparse + dense

    assert total.precision == pytest.approx(2 / 11)
    assert total.precision != pytest.approx((sparse.precision + dense.precision) / 2)


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        DetectionCounts(true_positives=-1)


def test_identical_landmarks_have_zero_error() -> None:
    assert normalized_landmark_error(REFERENCE, REFERENCE) == pytest.approx(0.0)


def test_error_is_normalized_by_interocular_distance() -> None:
    """Every point off by 2 px on a 20 px interocular face is an error of 0.1."""
    shifted = landmarks((42, 50), (62, 50), (52, 62), (44, 74), (60, 74))

    assert normalized_landmark_error(REFERENCE, shifted) == pytest.approx(0.1)


def test_the_same_pixel_error_on_a_smaller_face_scores_worse() -> None:
    """This is the whole point of normalizing: scale-free comparison across images."""
    small = landmarks((10, 20), (20, 20), (15, 26), (11, 32), (19, 32))
    small_shifted = landmarks((12, 20), (22, 20), (17, 26), (13, 32), (21, 32))
    big_shifted = landmarks((42, 50), (62, 50), (52, 62), (44, 74), (60, 74))

    small_error = normalized_landmark_error(small, small_shifted)
    big_error = normalized_landmark_error(REFERENCE, big_shifted)

    assert small_error == pytest.approx(0.2)
    assert big_error == pytest.approx(0.1)
    assert small_error > big_error


def test_error_averages_across_all_five_points() -> None:
    # Only the nose moves, by 5 px, over an interocular distance of 20.
    one_point_off = landmarks((40, 50), (60, 50), (55, 62), (42, 74), (58, 74))

    assert normalized_landmark_error(REFERENCE, one_point_off) == pytest.approx(5 / (5 * 20))


def test_degenerate_ground_truth_is_rejected_rather_than_scored() -> None:
    """Coincident eyes give nothing to normalize by; silently scoring zero would lie."""
    degenerate = landmarks((50, 50), (50, 50), (50, 62), (42, 74), (58, 74))

    with pytest.raises(ValueError, match="eye landmarks coincide"):
        normalized_landmark_error(degenerate, REFERENCE)
