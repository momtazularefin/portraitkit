"""Stage 2: ICAO-style geometry cropping.

The stage turns a detection result into a preset-sized portrait plus an assessment of the
geometry it achieved. Where correct framing needs canvas the source photograph does not
contain, the stage pads rather than quietly reframing to something that fits, and records
that it did. The legacy system this project succeeds handled that case by silently
shrinking the head, which produced pictures that looked fine and satisfied nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from portraitkit.crop.compliance import GeometryAssessment, assess_geometry
from portraitkit.crop.derotate import Derotation, level_eye_line
from portraitkit.crop.geometry import CropPlan, estimate_head, solve_crop
from portraitkit.crop.presets import DEFAULT_PRESET, CropPreset, get_preset
from portraitkit.imaging.io import LoadedImage
from portraitkit.types import DetectionResult, ImageSize

__all__ = ["CropConfig", "CropResult", "CropStage", "CropStatus"]


class CropStatus(StrEnum):
    """Outcome of the crop stage."""

    OK = "ok"
    NO_FACE = "no_face"
    """Detection found no subject, so there is nothing to frame."""

    NO_LANDMARKS = "no_landmarks"
    """The detector produced a box but no landmarks, leaving no geometry to work from."""

    PADDING_REQUIRED = "padding_required"
    """Correct framing needed canvas beyond the source and padding was not permitted."""

    DEGENERATE_LANDMARKS = "degenerate_landmarks"
    """Landmarks collapsed onto each other, leaving no scale to infer head extent from."""


@dataclass(frozen=True, slots=True)
class CropConfig:
    """Stage behaviour."""

    preset: str = DEFAULT_PRESET
    allow_padding: bool = True
    """Whether canvas may be added when the source photograph runs out."""

    background: tuple[int, int, int] = (255, 255, 255)
    """RGB fill used for added canvas."""

    derotate: bool = True
    """Level the eye line before solving geometry. A tilted head otherwise occupies more
    vertical extent than its true crown-to-chin length and is framed wrongly."""

    roll_tolerance_degrees: float = 1.0
    """Tilt at or below which levelling is skipped."""


@dataclass(frozen=True, slots=True)
class CropResult:
    """Typed result of the crop stage."""

    status: CropStatus
    preset: str
    image: np.ndarray | None = None
    """The produced crop as an ``(H, W, 3)`` uint8 RGB array, when one was produced."""

    plan: CropPlan | None = None
    assessment: GeometryAssessment | None = None
    padded: bool = False
    derotation: Derotation | None = None
    duration_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether a crop was produced."""
        return self.status is CropStatus.OK and self.image is not None

    @property
    def conforms(self) -> bool:
        """Whether every applicable geometry check passed.

        This means nothing PortraitKit can check is wrong. It is not a compliance
        certification; see :mod:`portraitkit.crop.compliance`.
        """
        return self.assessment is not None and self.assessment.conforms


def _extract(image: np.ndarray, plan: CropPlan, background: tuple[int, int, int]) -> np.ndarray:
    """Extract ``plan.rect`` from ``image``, padding where the rectangle overhangs."""
    height, width = image.shape[:2]
    x1, y1 = round(plan.rect.x1), round(plan.rect.y1)
    x2, y2 = round(plan.rect.x2), round(plan.rect.y2)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    inner = image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    if pad_left or pad_top or pad_right or pad_bottom:
        inner = cv2.copyMakeBorder(
            inner,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=tuple(int(channel) for channel in background),
        )
    return inner


def _resize(image: np.ndarray, target: ImageSize) -> np.ndarray:
    shrinking = target.width < image.shape[1]
    return cv2.resize(
        image,
        (target.width, target.height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC,
    )


class CropStage:
    """Produces a preset-sized portrait from a detection result."""

    def __init__(self, config: CropConfig | None = None) -> None:
        self.config = config or CropConfig()
        self.preset: CropPreset = get_preset(self.config.preset)

    def run(self, image: LoadedImage | np.ndarray, detection: DetectionResult) -> CropResult:
        """Crop ``image`` according to ``detection``.

        Args:
            image: The upright image the detection was produced from.
            detection: Stage 1 output.

        Returns:
            The crop and its geometry assessment. Every failure mode is a status rather
            than an exception, matching stage 1.
        """
        started = time.perf_counter()
        pixels = image.pixels if isinstance(image, LoadedImage) else np.asarray(image)

        def finish(status: CropStatus, **extra: object) -> CropResult:
            return CropResult(
                status=status,
                preset=self.preset.name,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                **extra,  # type: ignore[arg-type]
            )

        if detection.primary is None:
            return finish(CropStatus.NO_FACE)
        landmarks = detection.primary.landmarks
        if landmarks is None:
            return finish(CropStatus.NO_LANDMARKS)

        derotation: Derotation | None = None
        if self.config.derotate:
            pixels, landmarks, derotation = level_eye_line(
                pixels,
                landmarks,
                tolerance_degrees=self.config.roll_tolerance_degrees,
                background=self.config.background,
            )

        try:
            head = estimate_head(landmarks)
        except ValueError:
            return finish(CropStatus.DEGENERATE_LANDMARKS)

        source_size = ImageSize(width=int(pixels.shape[1]), height=int(pixels.shape[0]))
        plan = solve_crop(head, self.preset, source_size)

        if plan.needs_padding and not self.config.allow_padding:
            return finish(
                CropStatus.PADDING_REQUIRED,
                plan=plan,
                assessment=assess_geometry(plan),
                derotation=derotation,
            )

        cropped = _resize(_extract(pixels, plan, self.config.background), self.preset.output_size)
        return finish(
            CropStatus.OK,
            image=cropped,
            plan=plan,
            assessment=assess_geometry(plan),
            padded=plan.needs_padding,
            derotation=derotation,
            metadata={
                "dpi": self.preset.dpi,
                "output_size": list(self.preset.output_size.as_tuple()),
                "head_length_ratio": round(plan.achieved_head_length_ratio, 4),
                "face_centre_vertical": round(plan.achieved_face_centre_vertical, 4),
                "derotated_degrees": round(derotation.angle_degrees, 3) if derotation else 0.0,
            },
        )
