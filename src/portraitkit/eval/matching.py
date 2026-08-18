"""Matching predicted faces to ground truth.

Detection metrics are only as trustworthy as the rule that decides which prediction
corresponds to which annotated face. The rule here is the conventional one: walk
predictions from most to least confident, and give each the best-overlapping ground-truth
face still unclaimed, provided the overlap clears a threshold.

Processing in confidence order matters. If two predictions overlap the same face, the
confident one should claim it and the other should count as a false positive; matching in
arbitrary order could instead pair the weak prediction with the face and leave the strong
one unmatched, which flatters or punishes a detector for no reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from portraitkit.types import BoundingBox, FaceDetection

__all__ = ["Match", "MatchResult", "match_detections"]


@dataclass(frozen=True, slots=True)
class Match:
    """One accepted correspondence between a ground-truth face and a prediction."""

    truth_index: int
    prediction_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The full correspondence between one image's truth and predictions."""

    matches: tuple[Match, ...]
    unmatched_truth: tuple[int, ...]
    unmatched_predictions: tuple[int, ...]

    @property
    def true_positives(self) -> int:
        """Predictions that matched a ground-truth face."""
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        """Predictions that matched nothing."""
        return len(self.unmatched_predictions)

    @property
    def false_negatives(self) -> int:
        """Ground-truth faces no prediction claimed."""
        return len(self.unmatched_truth)

    def prediction_for(self, truth_index: int) -> int | None:
        """Return the prediction matched to ``truth_index``, if any."""
        for match in self.matches:
            if match.truth_index == truth_index:
                return match.prediction_index
        return None


def match_detections(
    truth: Sequence[BoundingBox],
    predictions: Sequence[FaceDetection],
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedily match ``predictions`` to ``truth`` by overlap, confidence first.

    Args:
        truth: Ground-truth boxes for one image.
        predictions: Detections for the same image, in any order.
        iou_threshold: Minimum overlap for a pair to count as a match.

    Returns:
        The matches plus the indices left unmatched on each side.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        msg = f"iou_threshold must be in [0, 1], got {iou_threshold}"
        raise ValueError(msg)

    order = sorted(range(len(predictions)), key=lambda index: -predictions[index].score)
    claimed: set[int] = set()
    matches: list[Match] = []
    unmatched_predictions: list[int] = []

    for prediction_index in order:
        box = predictions[prediction_index].box
        best_index = -1
        best_iou = 0.0
        for truth_index, truth_box in enumerate(truth):
            if truth_index in claimed:
                continue
            overlap = truth_box.intersection_over_union(box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = truth_index

        if best_index >= 0 and best_iou >= iou_threshold:
            claimed.add(best_index)
            matches.append(
                Match(truth_index=best_index, prediction_index=prediction_index, iou=best_iou)
            )
        else:
            unmatched_predictions.append(prediction_index)

    return MatchResult(
        matches=tuple(sorted(matches, key=lambda match: match.truth_index)),
        unmatched_truth=tuple(index for index in range(len(truth)) if index not in claimed),
        unmatched_predictions=tuple(sorted(unmatched_predictions)),
    )
