"""Matching predictions to ground truth."""

from __future__ import annotations

import pytest

from portraitkit.eval.matching import match_detections
from portraitkit.types import BoundingBox, FaceDetection


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def prediction(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> FaceDetection:
    return FaceDetection(box=box(x1, y1, x2, y2), score=score)


def test_no_truth_and_no_predictions_is_empty() -> None:
    result = match_detections([], [])

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_perfect_overlap_matches() -> None:
    result = match_detections([box(0, 0, 10, 10)], [prediction(0, 0, 10, 10)])

    assert result.true_positives == 1
    assert result.matches[0].iou == pytest.approx(1.0)
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_missed_face_counts_as_a_false_negative() -> None:
    result = match_detections([box(0, 0, 10, 10)], [])

    assert result.false_negatives == 1
    assert result.unmatched_truth == (0,)


def test_spurious_prediction_counts_as_a_false_positive() -> None:
    result = match_detections([], [prediction(0, 0, 10, 10)])

    assert result.false_positives == 1
    assert result.unmatched_predictions == (0,)


def test_overlap_below_threshold_is_not_a_match() -> None:
    # Intersection 50, union 150, IoU = 1/3.
    truth = [box(0, 0, 10, 10)]
    predictions = [prediction(5, 0, 15, 10)]

    assert match_detections(truth, predictions, iou_threshold=0.5).true_positives == 0
    assert match_detections(truth, predictions, iou_threshold=0.3).true_positives == 1


def test_each_truth_face_is_claimed_at_most_once() -> None:
    """A second overlapping prediction is a false positive, not a second match."""
    truth = [box(0, 0, 10, 10)]
    predictions = [prediction(0, 0, 10, 10, score=0.9), prediction(1, 1, 11, 11, score=0.8)]

    result = match_detections(truth, predictions)

    assert result.true_positives == 1
    assert result.false_positives == 1


def test_the_confident_prediction_claims_the_face() -> None:
    """Order of matching must follow confidence, not list order.

    If the weaker prediction claimed the face first, the stronger one would be scored as
    a false positive and the detector would be punished for being right.
    """
    truth = [box(0, 0, 10, 10)]
    weak = prediction(0, 0, 10, 10, score=0.4)
    strong = prediction(0, 0, 10, 10, score=0.95)

    result = match_detections(truth, [weak, strong])

    assert result.matches[0].prediction_index == 1
    assert result.unmatched_predictions == (0,)


def test_multiple_faces_match_independently() -> None:
    truth = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    predictions = [prediction(100, 100, 110, 110, score=0.8), prediction(0, 0, 10, 10, score=0.9)]

    result = match_detections(truth, predictions)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_matches_are_reported_in_truth_order() -> None:
    """Stable ordering keeps serialized reports diffable."""
    truth = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    predictions = [prediction(100, 100, 110, 110, score=0.99), prediction(0, 0, 10, 10, score=0.5)]

    result = match_detections(truth, predictions)

    assert [match.truth_index for match in result.matches] == [0, 1]


def test_prediction_lookup_by_truth_index() -> None:
    truth = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    predictions = [prediction(0, 0, 10, 10), prediction(100, 100, 110, 110)]

    result = match_detections(truth, predictions)

    assert result.prediction_for(1) == 1
    assert result.prediction_for(5) is None


def test_a_prediction_takes_its_best_overlap() -> None:
    truth = [box(0, 0, 10, 10), box(2, 2, 12, 12)]
    predictions = [prediction(2, 2, 12, 12)]

    result = match_detections(truth, predictions)

    assert result.matches[0].truth_index == 1
    assert result.matches[0].iou == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [-0.5, 1.1])
def test_out_of_range_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"iou_threshold must be in \[0, 1\]"):
        match_detections([], [], iou_threshold=threshold)
