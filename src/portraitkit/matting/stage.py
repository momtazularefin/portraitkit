"""Stage 3: background removal and replacement.

The stage processes an image through a declared matting adapter, returning a typed result
carrying the continuous alpha matte, the composite image (solid fill or transparency),
timing, and provenance diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portraitkit.imaging.io import LoadedImage, load_image
from portraitkit.matting.base import MattingAdapter
from portraitkit.types import Diagnostic, ImageSize

__all__ = ["MattingResult", "MattingStage", "MattingStageConfig", "composite_matte"]


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Parse a hex color string like '#ffffff' or 'ffffff' into an RGB tuple."""
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) == 6:
        return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
    if len(cleaned) == 3:
        return (int(cleaned[0] * 2, 16), int(cleaned[1] * 2, 16), int(cleaned[2] * 2, 16))
    msg = f"invalid hex color {hex_str!r}"
    raise ValueError(msg)


def parse_color(color: str | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    """Parse color names, hex codes, or RGB tuples into an RGB tuple or None."""
    if color is None:
        return None
    if isinstance(color, tuple):
        if len(color) != 3 or any(not 0 <= c <= 255 for c in color):
            msg = f"RGB color components must be in [0, 255], got {color}"
            raise ValueError(msg)
        return color
    lower = color.strip().lower()
    if lower in ("transparent", "none", "alpha", ""):
        return None
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "light_gray": (220, 220, 220),
        "light_grey": (220, 220, 220),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "blue": (0, 102, 204),
        "passport_blue": (30, 90, 180),
        "light_blue": (200, 225, 250),
        "red": (220, 20, 60),
    }
    if lower in named:
        return named[lower]
    return _hex_to_rgb(color)


@dataclass(frozen=True, slots=True)
class MattingStageConfig:
    """Configuration for the background stage."""

    background_color: tuple[int, int, int] | None = (255, 255, 255)
    """Background color to blend behind foreground. None produces transparent output."""

    threshold: float | None = None
    """Optional binarization threshold in [0, 1]. When provided, hardens alpha values."""

    def __post_init__(self) -> None:
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            msg = f"threshold must be in [0, 1], got {self.threshold}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MattingResult:
    """The outcome of a background-removal stage run."""

    alpha_matte: np.ndarray
    """``(H, W)`` float32 alpha matte with values in ``[0.0, 1.0]``."""

    image_rgb: np.ndarray
    """``(H, W, 3)`` uint8 image composited with background color."""

    image_rgba: np.ndarray
    """``(H, W, 4)`` uint8 image with foreground alpha channel."""

    image_size: ImageSize
    """Spatial dimensions of the processed image."""

    matter: str
    """Registry name of the model that produced this matte."""

    duration_ms: float
    """Inference and postprocessing duration in milliseconds."""

    diagnostics: tuple[Diagnostic, ...] = ()
    """Non-fatal diagnostic flags."""

    metadata: dict[str, Any] | None = None
    """Arbitrary stage metadata."""

    @property
    def ok(self) -> bool:
        return True


def composite_matte(
    image_rgb: np.ndarray,
    alpha_matte: np.ndarray,
    background_color: tuple[int, int, int] | None = (255, 255, 255),
    *,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite an RGB image with an alpha matte into RGB and RGBA arrays.

    Args:
        image_rgb: ``(H, W, 3)`` uint8 source image.
        alpha_matte: ``(H, W)`` float32 alpha matte in ``[0.0, 1.0]``.
        background_color: Solid RGB fill, or None for transparent black behind RGB.
        threshold: Optional threshold to binarize alpha values.

    Returns:
        A tuple of ``(image_rgb, image_rgba)``.
    """
    alpha = np.clip(alpha_matte, 0.0, 1.0).astype(np.float32)
    if threshold is not None:
        alpha = np.where(alpha >= threshold, 1.0, 0.0).astype(np.float32)

    alpha_3d = alpha[:, :, np.newaxis]
    fg = image_rgb.astype(np.float32)

    bg_rgb = (
        np.asarray(background_color, dtype=np.float32)
        if background_color is not None
        else np.asarray((255, 255, 255), dtype=np.float32)
    )

    rgb_composite = np.clip(fg * alpha_3d + bg_rgb * (1.0 - alpha_3d), 0, 255).astype(np.uint8)

    rgba_composite = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
    rgba_composite[:, :, :3] = image_rgb
    rgba_composite[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    return rgb_composite, rgba_composite


class MattingStage:
    """Runs a matting adapter and composites foreground/background."""

    def __init__(self, matter: MattingAdapter, config: MattingStageConfig | None = None) -> None:
        self.matter = matter
        self.config = config or MattingStageConfig()

    def run(self, image: LoadedImage | np.ndarray | str | Path) -> MattingResult:
        """Process an image, extract alpha, and generate composited outputs.

        Args:
            image: Loaded image object, uint8 RGB numpy array, or path to load.

        Returns:
            A typed :class:`MattingResult`.
        """
        loaded = image if isinstance(image, LoadedImage) else None
        if isinstance(image, str | Path):
            loaded = load_image(image)
        pixels = loaded.pixels if loaded is not None else np.asarray(image)

        started = time.perf_counter()
        alpha = self.matter.predict_alpha(pixels)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        rgb_out, rgba_out = composite_matte(
            pixels,
            alpha,
            background_color=self.config.background_color,
            threshold=self.config.threshold,
        )

        size = ImageSize(width=int(pixels.shape[1]), height=int(pixels.shape[0]))
        diagnostics: list[Diagnostic] = []
        if loaded is not None:
            if loaded.orientation.applied:
                diagnostics.append(Diagnostic.ORIENTATION_CORRECTED)
            if loaded.truncated:
                diagnostics.append(Diagnostic.TRUNCATED_IMAGE_DATA)

        return MattingResult(
            alpha_matte=alpha,
            image_rgb=rgb_out,
            image_rgba=rgba_out,
            image_size=size,
            matter=self.matter.name,
            duration_ms=elapsed_ms,
            diagnostics=tuple(diagnostics),
            metadata={
                "background_color": self.config.background_color,
                "threshold": self.config.threshold,
                "providers": list(self.matter.info.providers),
            },
        )
