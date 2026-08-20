"""Parameterized image degradation suite for robustness benchmarking.

Simulates real-world portrait capture defects (compression artifacts, motion blur,
poor lighting, sensor noise, downsampling, and background clutter) deterministically.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "ClutteredBackground",
    "Degradation",
    "Downscale",
    "GaussianBlur",
    "GaussianNoise",
    "JpegCompression",
    "LowLight",
    "MotionBlur",
    "apply_degradations",
    "build_degradation",
]


class Degradation(ABC):
    """Abstract base for a parameterized image corruption."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the corruption type."""

    @property
    @abstractmethod
    def severity(self) -> str:
        """Human-readable severity level."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the degradation to an ``(H, W, 3)`` uint8 RGB image.

        Args:
            image: Source uint8 RGB array.

        Returns:
            Degraded ``(H, W, 3)`` uint8 RGB array with identical dimensions.
        """

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "severity": self.severity}


@dataclass(frozen=True, slots=True)
class JpegCompression(Degradation):
    """JPEG compression artifacts."""

    quality: int = 50

    def __post_init__(self) -> None:
        if not 1 <= self.quality <= 100:
            msg = f"JPEG quality must be in [1, 100], got {self.quality}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "jpeg_compression"

    @property
    def severity(self) -> str:
        return f"q{self.quality}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image, mode="RGB")
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        decoded = Image.open(buffer)
        return np.asarray(decoded, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class GaussianBlur(Degradation):
    """Gaussian defocus blur."""

    sigma: float = 2.0
    kernel_size: int | None = None

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            msg = f"sigma must be positive, got {self.sigma}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "gaussian_blur"

    @property
    def severity(self) -> str:
        return f"sigma_{self.sigma:g}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        ksize = self.kernel_size
        if ksize is None:
            # 2 * round(3 * sigma) + 1
            k = int(2 * round(3.0 * self.sigma) + 1)
            ksize = max(3, k)
            if ksize % 2 == 0:
                ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), self.sigma)


@dataclass(frozen=True, slots=True)
class MotionBlur(Degradation):
    """Linear directional motion blur."""

    kernel_size: int = 15
    angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        if self.kernel_size < 3:
            msg = f"kernel_size must be >= 3, got {self.kernel_size}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "motion_blur"

    @property
    def severity(self) -> str:
        return f"k{self.kernel_size}_a{self.angle_degrees:g}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size, dtype=np.float32)
        # Rotate kernel
        rot_mat = cv2.getRotationMatrix2D(
            (self.kernel_size / 2 - 0.5, self.kernel_size / 2 - 0.5),
            self.angle_degrees,
            1.0,
        )
        kernel = cv2.warpAffine(kernel, rot_mat, (self.kernel_size, self.kernel_size))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        return cv2.filter2D(image, -1, kernel)


@dataclass(frozen=True, slots=True)
class LowLight(Degradation):
    """Low-light / underexposure and non-linear contrast reduction."""

    factor: float = 0.4
    gamma: float = 1.2

    def __post_init__(self) -> None:
        if self.factor <= 0.0 or self.gamma <= 0.0:
            msg = f"factor and gamma must be positive, got factor={self.factor}, gamma={self.gamma}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "low_light"

    @property
    def severity(self) -> str:
        return f"f{self.factor:g}_g{self.gamma:g}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        norm = (image.astype(np.float32) / 255.0) * self.factor
        if self.gamma != 1.0:
            norm = np.power(norm, self.gamma)
        return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class GaussianNoise(Degradation):
    """Additive sensor Gaussian noise."""

    std: float = 25.0
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.std < 0.0:
            msg = f"std must be non-negative, got {self.std}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "gaussian_noise"

    @property
    def severity(self) -> str:
        return f"std_{self.std:g}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class Downscale(Degradation):
    """Low-resolution downscaling and bilinear upsampling."""

    scale: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.scale < 1.0:
            msg = f"scale must be in (0, 1), got {self.scale}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "downscale"

    @property
    def severity(self) -> str:
        return f"x{self.scale:g}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small_w = max(1, round(w * self.scale))
        small_h = max(1, round(h * self.scale))
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True, slots=True)
class ClutteredBackground(Degradation):
    """Synthetic background high-frequency clutter injection."""

    pattern: str = "checkerboard"
    frequency: int = 16

    @property
    def name(self) -> str:
        return "cluttered_background"

    @property
    def severity(self) -> str:
        return f"{self.pattern}_{self.frequency}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Construct synthetic high-frequency texture
        x = np.arange(w) // self.frequency
        y = np.arange(h) // self.frequency
        grid = (x[np.newaxis, :] + y[:, np.newaxis]) % 2
        texture = (grid * 180 + 40).astype(np.uint8)[:, :, np.newaxis]
        texture_3ch = np.repeat(texture, 3, axis=2)
        # Blend slightly with borders of image (50% blend in outer 15% margin)
        mask = np.zeros((h, w, 1), dtype=np.float32)
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        mask[:margin_h, :] = 0.5
        mask[h - margin_h :, :] = 0.5
        mask[:, :margin_w] = 0.5
        mask[:, w - margin_w :] = 0.5

        out = image.astype(np.float32) * (1.0 - mask) + texture_3ch.astype(np.float32) * mask
        return np.clip(out, 0, 255).astype(np.uint8)


def build_degradation(spec: dict[str, Any]) -> Degradation:
    """Build a degradation instance from a dictionary specification."""
    kind = spec.get("type") or spec.get("name")
    if not kind:
        msg = f"degradation spec missing 'type': {spec}"
        raise ValueError(msg)

    kind_lower = kind.lower().replace("-", "_")
    params = {k: v for k, v in spec.items() if k not in ("type", "name")}

    if kind_lower in ("jpeg", "jpeg_compression", "compression"):
        return JpegCompression(**params)
    if kind_lower in ("blur", "gaussian_blur", "defocus"):
        return GaussianBlur(**params)
    if kind_lower in ("motion", "motion_blur"):
        return MotionBlur(**params)
    if kind_lower in ("low_light", "underexposure", "dark"):
        return LowLight(**params)
    if kind_lower in ("noise", "gaussian_noise", "sensor_noise"):
        return GaussianNoise(**params)
    if kind_lower in ("downscale", "low_res", "resolution"):
        return Downscale(**params)
    if kind_lower in ("clutter", "cluttered_background", "background"):
        return ClutteredBackground(**params)

    msg = f"unknown degradation type {kind!r}"
    raise ValueError(msg)


def apply_degradations(image: np.ndarray, degradations: Sequence[Degradation]) -> np.ndarray:
    """Sequentially apply degradations to an image."""
    current = image
    for deg in degradations:
        current = deg.apply(current)
    return current
