"""Stage 1: detection and orientation.

The stage takes an arbitrary photo and returns a typed result: the faces found, the
primary subject, and the diagnostics that explain what made the image easy or hard. It
does not raise when a photo contains no face -- that is an ordinary outcome with its own
status, not an exceptional one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from portraitkit.config import Settings
from portraitkit.detection.base import DetectorConfig, FaceDetector
from portraitkit.detection.scrfd import ScrfdDetector
from portraitkit.detection.selection import SelectionStrategy, select_primary
from portraitkit.detection.yunet import YuNetDetector
from portraitkit.errors import ModelError
from portraitkit.imaging.io import LoadedImage, load_image
from portraitkit.models.registry import DEFAULT_DETECTOR, get_model
from portraitkit.models.session import CPU_PROVIDER
from portraitkit.models.store import resolve_model
from portraitkit.types import (
    DetectionResult,
    DetectionStatus,
    Diagnostic,
    FaceDetection,
    ImageSize,
)

__all__ = ["DetectionStage", "StageConfig", "build_detector"]

_ADAPTERS: dict[str, type[FaceDetector]] = {
    "yunet-2023mar": YuNetDetector,
    "scrfd-10g-bnkps": ScrfdDetector,
}


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Stage behaviour beyond the detector's own thresholds."""

    selection: SelectionStrategy = SelectionStrategy.LARGEST
    """Rule for choosing the primary subject."""

    low_confidence_below: float = 0.8
    """Primary-face confidence under which the result is flagged as uncertain."""

    small_face_ratio: float = 0.05
    """Flag the primary face when its height is a smaller fraction of the frame than
    this. A subject this distant rarely survives an ICAO head-height crop."""

    border_margin_px: float = 2.0
    """Distance from an edge within which the primary face counts as touching it."""

    strong_roll_degrees: float = 15.0
    """Absolute eye-line tilt above which the head is flagged as noticeably rotated."""


def build_detector(
    model: str = DEFAULT_DETECTOR,
    *,
    settings: Settings | None = None,
    config: DetectorConfig | None = None,
    model_path: str | Path | None = None,
    providers: tuple[str, ...] = (CPU_PROVIDER,),
    allow_download: bool | None = None,
) -> FaceDetector:
    """Construct the adapter for ``model``, resolving its artifact if needed.

    Args:
        model: Registry name of the detector.
        settings: Configuration used to locate the model cache.
        config: Detector thresholds.
        model_path: Bypass resolution and load this file directly.
        providers: Execution providers, CPU by default.
        allow_download: Override the configured download policy.

    Raises:
        ModelError: If no adapter is registered for ``model``.
    """
    try:
        adapter = _ADAPTERS[model]
    except KeyError:
        known = ", ".join(sorted(_ADAPTERS))
        msg = f"no detector adapter for {model!r}; available adapters are: {known}"
        raise ModelError(msg) from None

    if model_path is None:
        spec = get_model(model)
        model_path = resolve_model(spec, settings, allow_download=allow_download)
    return adapter(model_path, config=config, providers=providers)


class DetectionStage:
    """Runs a detector over an image and reports a typed, diagnosed result."""

    def __init__(self, detector: FaceDetector, config: StageConfig | None = None) -> None:
        self.detector = detector
        self.config = config or StageConfig()

    def run(self, image: LoadedImage | np.ndarray | str | Path) -> DetectionResult:
        """Detect faces and select a primary subject.

        Args:
            image: A loaded image, an upright RGB array, or a path to load. Passing a
                path or a :class:`LoadedImage` also carries orientation and decode
                provenance into the diagnostics.

        Returns:
            The stage result. ``status`` is ``NO_FACE`` when nothing passed the
            detector's score threshold.
        """
        loaded = image if isinstance(image, LoadedImage) else None
        if isinstance(image, str | Path):
            loaded = load_image(image)
        pixels = loaded.pixels if loaded is not None else np.asarray(image)

        started = time.perf_counter()
        faces = self.detector.detect(pixels)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        size = ImageSize(width=int(pixels.shape[1]), height=int(pixels.shape[0]))
        primary = select_primary(faces, size, self.config.selection)
        diagnostics = self._diagnose(faces, primary, size, loaded)

        return DetectionResult(
            status=DetectionStatus.OK if primary is not None else DetectionStatus.NO_FACE,
            image_size=size,
            faces=faces,
            primary=primary,
            diagnostics=diagnostics,
            detector=self.detector.name,
            duration_ms=elapsed_ms,
            metadata={
                "selection": str(self.config.selection),
                "providers": list(self.detector.info.providers),
            },
        )

    def _diagnose(
        self,
        faces: tuple[FaceDetection, ...],
        primary: FaceDetection | None,
        size: ImageSize,
        loaded: LoadedImage | None,
    ) -> tuple[Diagnostic, ...]:
        """Collect the non-fatal observations that explain this result."""
        found: list[Diagnostic] = []

        if loaded is not None:
            if loaded.orientation.applied:
                found.append(Diagnostic.ORIENTATION_CORRECTED)
            if loaded.truncated:
                found.append(Diagnostic.TRUNCATED_IMAGE_DATA)

        if len(faces) > 1:
            found.append(Diagnostic.MULTIPLE_FACES)

        if primary is not None:
            if primary.score < self.config.low_confidence_below:
                found.append(Diagnostic.LOW_CONFIDENCE)
            if primary.box.height < self.config.small_face_ratio * size.height:
                found.append(Diagnostic.SMALL_FACE)

            margin = self.config.border_margin_px
            box = primary.box
            if (
                box.x1 <= margin
                or box.y1 <= margin
                or box.x2 >= size.width - margin
                or box.y2 >= size.height - margin
            ):
                found.append(Diagnostic.FACE_TOUCHES_BORDER)

            landmarks = primary.landmarks
            if (
                landmarks is not None
                and abs(landmarks.roll_degrees) > self.config.strong_roll_degrees
            ):
                found.append(Diagnostic.STRONG_ROLL)

        return tuple(found)
