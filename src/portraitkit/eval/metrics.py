"""Detection and landmark metrics.

Counts aggregate across images before ratios are taken, so a set of small images cannot
outvote a set of large ones the way averaging per-image precision would.
"""

from __future__ import annotations

from dataclasses import dataclass

from portraitkit.types import FaceLandmarks5

__all__ = ["DetectionCounts", "normalized_landmark_error"]


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    """Confusion counts for detection, summable across images.

    Ratios return ``0.0`` when their denominator is zero. A detector that predicted
    nothing has no precision to report, and propagating a null through every aggregate
    would cost more than it explains; the accompanying counts always make the empty case
    visible.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("true_positives", self.true_positives),
            ("false_positives", self.false_positives),
            ("false_negatives", self.false_negatives),
        ):
            if value < 0:
                msg = f"{label} cannot be negative, got {value}"
                raise ValueError(msg)

    def __add__(self, other: DetectionCounts) -> DetectionCounts:
        return DetectionCounts(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    @property
    def predicted(self) -> int:
        """Total predictions made."""
        return self.true_positives + self.false_positives

    @property
    def actual(self) -> int:
        """Total ground-truth faces."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        """Share of predictions that were correct."""
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        """Share of ground-truth faces that were found."""
        return self.true_positives / self.actual if self.actual else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2.0 * self.precision * self.recall / total if total else 0.0


def normalized_landmark_error(truth: FaceLandmarks5, predicted: FaceLandmarks5) -> float:
    """Mean landmark distance divided by the ground-truth interocular distance.

    Normalizing by the distance between the eyes is what makes the number comparable
    across images: a five-pixel error on a passport-sized face and on a thumbnail are not
    the same mistake, and a raw pixel average would rank them as though they were.

    Raises:
        ValueError: If the ground-truth eyes coincide, leaving nothing to normalize by.
    """
    normalizer = truth.interocular_distance
    if normalizer <= 0.0:
        msg = "cannot normalize landmark error: ground-truth eye landmarks coincide"
        raise ValueError(msg)

    total = sum(
        expected.distance_to(actual)
        for expected, actual in zip(truth.as_points(), predicted.as_points(), strict=True)
    )
    return total / (len(truth.as_points()) * normalizer)
