"""Evaluation reports and their deterministic serialization.

A report has to be diffable. When a detector changes, a reviewer should see which numbers
moved, not a reshuffled document. Serialization therefore sorts nothing arbitrarily,
keeps images in manifest order, and rounds every float to a fixed precision so that
platform-level differences in the last bits of a double do not show up as changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portraitkit.eval.metrics import DetectionCounts

__all__ = ["ROUND_DIGITS", "EvaluationReport", "ImageEvaluation"]

ROUND_DIGITS = 6
"""Decimal places retained when serializing. Well beyond metric significance, and short
enough that two runs of the same configuration produce byte-identical output."""


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True, slots=True)
class ImageEvaluation:
    """Per-image outcome."""

    path: str
    counts: DetectionCounts
    match_ious: tuple[float, ...] = ()
    landmark_errors: tuple[float, ...] = ()
    primary_correct: bool | None = None
    """Whether the selected primary subject matched the annotated one. ``None`` when the
    annotation declares no primary, or when no primary could be selected."""

    duration_ms: float = 0.0
    diagnostics: tuple[str, ...] = ()

    @property
    def mean_iou(self) -> float | None:
        """Mean overlap across this image's matches."""
        return _mean(list(self.match_ious))

    @property
    def mean_landmark_error(self) -> float | None:
        """Mean normalized landmark error across this image's matches."""
        return _mean(list(self.landmark_errors))

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "path": self.path,
            "true_positives": self.counts.true_positives,
            "false_positives": self.counts.false_positives,
            "false_negatives": self.counts.false_negatives,
            "mean_iou": None if self.mean_iou is None else _round(self.mean_iou),
            "mean_landmark_error": (
                None if self.mean_landmark_error is None else _round(self.mean_landmark_error)
            ),
            "primary_correct": self.primary_correct,
            "duration_ms": _round(self.duration_ms),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A complete evaluation run: what was measured, how, and what came out."""

    dataset: str
    detector: str
    settings: dict[str, Any]
    """Everything needed to reproduce the run: thresholds, selection strategy, providers."""

    images: tuple[ImageEvaluation, ...] = ()
    errors: tuple[tuple[str, str], ...] = ()
    """Images that could not be evaluated, as ``(path, reason)`` pairs. Recorded rather
    than dropped, so a run that skipped half its inputs cannot look like a clean sweep."""

    counts: DetectionCounts = field(default_factory=DetectionCounts)

    @property
    def evaluated_images(self) -> int:
        """Images that produced a result."""
        return len(self.images)

    @property
    def mean_iou(self) -> float | None:
        """Mean overlap across every match in the run."""
        return _mean([iou for image in self.images for iou in image.match_ious])

    @property
    def mean_landmark_error(self) -> float | None:
        """Mean normalized landmark error across every comparable match."""
        return _mean([error for image in self.images for error in image.landmark_errors])

    @property
    def primary_accuracy(self) -> float | None:
        """Share of images where the selected subject was the annotated one.

        ``None`` when no image in the set declares a primary subject, which keeps an
        unmeasured quantity from reading as a perfect score.
        """
        judged = [
            image.primary_correct for image in self.images if image.primary_correct is not None
        ]
        return sum(judged) / len(judged) if judged else None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, stable across runs of the same configuration."""
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "detector": self.detector,
            "settings": self.settings,
            "summary": {
                "evaluated_images": self.evaluated_images,
                "errored_images": len(self.errors),
                "true_positives": self.counts.true_positives,
                "false_positives": self.counts.false_positives,
                "false_negatives": self.counts.false_negatives,
                "precision": _round(self.counts.precision),
                "recall": _round(self.counts.recall),
                "f1": _round(self.counts.f1),
                "mean_iou": None if self.mean_iou is None else _round(self.mean_iou),
                "mean_landmark_error": (
                    None if self.mean_landmark_error is None else _round(self.mean_landmark_error)
                ),
                "primary_accuracy": (
                    None if self.primary_accuracy is None else _round(self.primary_accuracy)
                ),
            },
            "images": [image.to_dict() for image in self.images],
            "errors": [{"path": path, "reason": reason} for path, reason in self.errors],
        }

    def to_json(self) -> str:
        """Pretty-printed JSON, newline-terminated for clean diffs."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to ``path``, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return target
