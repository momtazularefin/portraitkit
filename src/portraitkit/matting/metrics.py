"""The four standard alpha-matting metrics.

SAD, MSE, gradient error, and connectivity error are the measures the matting literature
reports, introduced by the perceptually motivated benchmark of Rhemann et al. (2009).
PortraitBench implements them here rather than importing one model author's copy, for the
same reason non-maximum suppression lives in this project: if one entrant were scored by
its own toolkit and another by a different one, part of the measured difference would be
the scoring code.

Alpha values are floats in ``[0, 1]``. Where a trimap is supplied, only its unknown region
is evaluated, which is the convention the benchmark uses -- scoring the known foreground
and background would reward a model for copying pixels it was handed.

Scaling follows the literature: SAD, gradient, and connectivity are reported in thousands
so their magnitudes stay readable, while MSE is a plain per-pixel mean. The raw sums are
available too, because a scale convention should never be something a reader has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

__all__ = [
    "GRADIENT_SIGMA",
    "MattingMetrics",
    "connectivity_error",
    "gradient_error",
    "matting_metrics",
    "mean_squared_error",
    "sum_absolute_difference",
]

SCALE: Final = 1000.0
"""Divisor applied to summed metrics, as the matting literature reports them."""

GRADIENT_SIGMA: Final = 1.4
"""Standard deviation of the Gaussian derivative used for the gradient metric."""

CONNECTIVITY_STEP: Final = 0.1
"""Threshold step when tracing connectivity."""

CONNECTIVITY_THETA: Final = 0.15
"""Degree of connectivity below which a pixel contributes no error."""


@dataclass(frozen=True, slots=True)
class MattingMetrics:
    """The four benchmark metrics for one predicted matte. Lower is better throughout."""

    sad: float
    """Sum of absolute differences, in thousands."""

    mse: float
    """Mean squared error per evaluated pixel."""

    gradient: float
    """Gradient error, in thousands. Penalises smoothed or jagged edges that SAD misses."""

    connectivity: float
    """Connectivity error, in thousands. Penalises detached fragments."""

    evaluated_pixels: int
    """How many pixels the trimap admitted, so a reader can weight the numbers."""

    def to_dict(self) -> dict[str, float | int]:
        """Serializable form, rounded for stable reports."""
        return {
            "sad": round(self.sad, 6),
            "mse": round(self.mse, 8),
            "gradient": round(self.gradient, 6),
            "connectivity": round(self.connectivity, 6),
            "evaluated_pixels": self.evaluated_pixels,
        }


def _validate(predicted: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if predicted.shape != truth.shape:
        msg = f"matte shapes must match, got {predicted.shape} and {truth.shape}"
        raise ValueError(msg)
    if predicted.ndim != 2:
        msg = f"mattes must be two-dimensional, got {predicted.ndim} dimensions"
        raise ValueError(msg)
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    for label, matte in (("predicted", left), ("truth", right)):
        if matte.size and (matte.min() < 0.0 or matte.max() > 1.0):
            msg = f"{label} alpha must lie in [0, 1], got [{matte.min()}, {matte.max()}]"
            raise ValueError(msg)
    return left, right


def _region(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """Return the boolean evaluation region, defaulting to every pixel."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape != shape:
        msg = f"trimap shape {mask.shape} does not match the matte shape {shape}"
        raise ValueError(msg)
    return np.asarray(mask, dtype=bool)


def sum_absolute_difference(
    predicted: np.ndarray, truth: np.ndarray, unknown: np.ndarray | None = None
) -> float:
    """Summed absolute alpha error over the evaluated region, in thousands."""
    left, right = _validate(predicted, truth)
    region = _region(unknown, left.shape)
    return float(np.abs(left - right)[region].sum() / SCALE)


def mean_squared_error(
    predicted: np.ndarray, truth: np.ndarray, unknown: np.ndarray | None = None
) -> float:
    """Mean squared alpha error over the evaluated region."""
    left, right = _validate(predicted, truth)
    region = _region(unknown, left.shape)
    count = int(region.sum())
    if count == 0:
        return 0.0
    return float((np.square(left - right)[region]).sum() / count)


def _gaussian_gradient(matte: np.ndarray, sigma: float) -> np.ndarray:
    """Gradient magnitude from a first-order Gaussian derivative."""
    radius = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    smooth = np.exp(-(offsets**2) / (2.0 * sigma**2))
    smooth /= smooth.sum()
    derivative = -(offsets / (sigma**2)) * smooth

    horizontal = cv2.sepFilter2D(matte, cv2.CV_64F, derivative, smooth)
    vertical = cv2.sepFilter2D(matte, cv2.CV_64F, smooth, derivative)
    return np.sqrt(np.square(horizontal) + np.square(vertical))


def gradient_error(
    predicted: np.ndarray,
    truth: np.ndarray,
    unknown: np.ndarray | None = None,
    *,
    sigma: float = GRADIENT_SIGMA,
) -> float:
    """Squared difference of gradient magnitudes, in thousands.

    This is the metric that separates a matte with soft, plausible hair edges from one
    that scores similarly on SAD while being visibly smeared, because a constant alpha
    offset changes SAD but leaves the gradient untouched.
    """
    left, right = _validate(predicted, truth)
    region = _region(unknown, left.shape)
    difference = _gaussian_gradient(left, sigma) - _gaussian_gradient(right, sigma)
    return float(np.square(difference)[region].sum() / SCALE)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of ``mask`` as a boolean array."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # label 0 is background
    return labels == int(sizes.argmax())


def connectivity_error(
    predicted: np.ndarray,
    truth: np.ndarray,
    unknown: np.ndarray | None = None,
    *,
    step: float = CONNECTIVITY_STEP,
    theta: float = CONNECTIVITY_THETA,
) -> float:
    """Connectivity error, in thousands.

    Sweeps a threshold upward and records, for each pixel, the highest level at which it
    still belongs to the largest region both mattes agree on. A pixel that detaches early
    is penalised, which is what catches a matte that drops an ear or leaves a floating
    speck of background while scoring well elsewhere.
    """
    left, right = _validate(predicted, truth)
    region = _region(unknown, left.shape)

    thresholds = np.arange(0.0, 1.0 + step, step)
    connected_at = np.full(left.shape, -1.0, dtype=np.float64)
    for index in range(1, len(thresholds)):
        level = thresholds[index]
        shared = _largest_component((left >= level) & (right >= level))
        detaching = (connected_at < 0.0) & ~shared
        connected_at[detaching] = thresholds[index - 1]
    connected_at[connected_at < 0.0] = 1.0

    def degree(matte: np.ndarray) -> np.ndarray:
        distance = matte - connected_at
        return 1.0 - distance * (distance >= theta)

    return float(np.abs(degree(left) - degree(right))[region].sum() / SCALE)


def matting_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    unknown: np.ndarray | None = None,
    *,
    sigma: float = GRADIENT_SIGMA,
) -> MattingMetrics:
    """Compute all four benchmark metrics for one predicted matte.

    Args:
        predicted: ``(H, W)`` predicted alpha in ``[0, 1]``.
        truth: ``(H, W)`` ground-truth alpha in ``[0, 1]``.
        unknown: Optional boolean mask of the trimap's unknown region. When omitted every
            pixel is evaluated.
        sigma: Gaussian derivative width for the gradient metric.
    """
    left, right = _validate(predicted, truth)
    region = _region(unknown, left.shape)
    return MattingMetrics(
        sad=sum_absolute_difference(left, right, region),
        mse=mean_squared_error(left, right, region),
        gradient=gradient_error(left, right, region, sigma=sigma),
        connectivity=connectivity_error(left, right, region),
        evaluated_pixels=int(region.sum()),
    )
