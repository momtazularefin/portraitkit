"""Running matting evaluation over an annotation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from portraitkit.errors import ImageLoadError
from portraitkit.eval.matting_annotations import MattingAnnotationSet
from portraitkit.imaging.io import load_image
from portraitkit.matting.base import MattingAdapter
from portraitkit.matting.metrics import MattingMetrics, matting_metrics
from portraitkit.matting.stage import MattingStage
from portraitkit.types import ImageSize

__all__ = [
    "MattingEvaluation",
    "MattingEvaluationReport",
    "MattingEvaluationSummary",
    "evaluate_matting",
]


@dataclass(frozen=True, slots=True)
class MattingEvaluation:
    """Evaluation outcomes for one image sample."""

    path: str
    metrics: MattingMetrics
    duration_ms: float
    image_size: ImageSize

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "metrics": self.metrics.to_dict(),
            "duration_ms": round(self.duration_ms, 3),
            "image_size": list(self.image_size.as_tuple()),
        }


@dataclass(frozen=True, slots=True)
class MattingEvaluationSummary:
    """Aggregate dataset-level matting metrics."""

    evaluated_samples: int
    errored_samples: int
    mean_sad: float
    mean_mse: float
    mean_gradient: float
    mean_connectivity: float
    median_sad: float
    median_mse: float
    median_gradient: float
    median_connectivity: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "evaluated_samples": self.evaluated_samples,
            "errored_samples": self.errored_samples,
            "mean_sad": round(self.mean_sad, 6),
            "mean_mse": round(self.mean_mse, 8),
            "mean_gradient": round(self.mean_gradient, 6),
            "mean_connectivity": round(self.mean_connectivity, 6),
            "median_sad": round(self.median_sad, 6),
            "median_mse": round(self.median_mse, 8),
            "median_gradient": round(self.median_gradient, 6),
            "median_connectivity": round(self.median_connectivity, 6),
        }


@dataclass(frozen=True, slots=True)
class MattingEvaluationReport:
    """Complete evaluation report over a matting dataset."""

    dataset: str
    matter: str
    summary: MattingEvaluationSummary
    samples: tuple[MattingEvaluation, ...]
    errors: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "matter": self.matter,
            "summary": self.summary.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "errors": [{"path": path, "error": error} for path, error in self.errors],
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _load_mask(path: Path) -> np.ndarray:
    """Load an alpha mask or trimap as a 2D float array in [0, 1]."""
    if not path.exists():
        msg = f"alpha mask file {str(path)!r} does not exist"
        raise ImageLoadError(msg)
    mask_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        msg = f"could not decode mask {str(path)!r}"
        raise ImageLoadError(msg)
    return mask_img.astype(np.float64) / 255.0


def evaluate_matting(
    matter_or_stage: MattingAdapter | MattingStage,
    annotations: MattingAnnotationSet,
) -> MattingEvaluationReport:
    """Evaluate a matting adapter or stage against ground truth annotations.

    Args:
        matter_or_stage: The matting model or stage to grade.
        annotations: Ground-truth dataset manifest.

    Returns:
        A structured :class:`MattingEvaluationReport`.
    """
    matter = (
        matter_or_stage.matter if isinstance(matter_or_stage, MattingStage) else matter_or_stage
    )
    evaluations: list[MattingEvaluation] = []
    errors: list[tuple[str, str]] = []

    for sample in annotations.samples:
        try:
            image = load_image(sample.image_path)
            truth_alpha = _load_mask(sample.alpha_path)
        except ImageLoadError as error:
            errors.append((sample.relative_image, str(error)))
            continue

        if isinstance(matter_or_stage, MattingStage):
            result = matter_or_stage.run(image)
            pred_alpha = result.alpha_matte
            duration_ms = result.duration_ms
        else:
            started = cv2.getTickCount()
            pred_alpha = matter_or_stage.predict_alpha(image.pixels)
            duration_ms = (cv2.getTickCount() - started) / cv2.getTickFrequency() * 1000.0

        # Check dimension alignment
        if pred_alpha.shape != truth_alpha.shape:
            # Resize pred_alpha or truth_alpha if slight dimension mismatch
            truth_alpha = cv2.resize(
                truth_alpha,
                (pred_alpha.shape[1], pred_alpha.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        unknown_mask = None
        if sample.trimap_path and sample.trimap_path.exists():
            try:
                trimap = cv2.imread(str(sample.trimap_path), cv2.IMREAD_GRAYSCALE)
                if trimap is not None:
                    # Conventional trimap: 0=bg, 255=fg, 128=unknown (or 0 < val < 255)
                    unknown_mask = (trimap > 0) & (trimap < 255)
            except Exception:
                unknown_mask = None

        metrics = matting_metrics(pred_alpha, truth_alpha, unknown=unknown_mask)

        evaluations.append(
            MattingEvaluation(
                path=sample.relative_image,
                metrics=metrics,
                duration_ms=duration_ms,
                image_size=ImageSize(width=pred_alpha.shape[1], height=pred_alpha.shape[0]),
            )
        )

    if evaluations:
        sads = [e.metrics.sad for e in evaluations]
        mses = [e.metrics.mse for e in evaluations]
        grads = [e.metrics.gradient for e in evaluations]
        conns = [e.metrics.connectivity for e in evaluations]
        summary = MattingEvaluationSummary(
            evaluated_samples=len(evaluations),
            errored_samples=len(errors),
            mean_sad=float(np.mean(sads)),
            mean_mse=float(np.mean(mses)),
            mean_gradient=float(np.mean(grads)),
            mean_connectivity=float(np.mean(conns)),
            median_sad=float(np.median(sads)),
            median_mse=float(np.median(mses)),
            median_gradient=float(np.median(grads)),
            median_connectivity=float(np.median(conns)),
        )
    else:
        summary = MattingEvaluationSummary(
            evaluated_samples=0,
            errored_samples=len(errors),
            mean_sad=0.0,
            mean_mse=0.0,
            mean_gradient=0.0,
            mean_connectivity=0.0,
            median_sad=0.0,
            median_mse=0.0,
            median_gradient=0.0,
            median_connectivity=0.0,
        )

    return MattingEvaluationReport(
        dataset=annotations.name,
        matter=matter.name,
        summary=summary,
        samples=tuple(evaluations),
        errors=tuple(errors),
    )
