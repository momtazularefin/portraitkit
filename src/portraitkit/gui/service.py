"""Analysis service behind the inspector UI.

Deliberately free of HTTP. Everything here takes bytes and returns plain data, so the
whole surface is unit-testable without opening a socket, and the server module stays a
thin adapter. This is the same split the CLI uses against the stages.

The service exists to make the pipeline's *measurements* visible, not merely its output.
A portrait tool that shows only the finished crop hides the part that matters: which
constraints were checked, what each one measured, and how much the answer can be trusted.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from portraitkit.config import load_settings
from portraitkit.crop.compliance import CheckBasis, CheckStatus
from portraitkit.crop.presets import DEFAULT_PRESET, PRESETS
from portraitkit.crop.stage import CropConfig, CropStage
from portraitkit.detection.stage import DetectionStage, build_detector
from portraitkit.errors import PortraitKitError
from portraitkit.imaging.io import LoadedImage, load_image
from portraitkit.models.registry import MODELS
from portraitkit.types import DetectionResult

__all__ = ["AnalysisService", "InspectorOptions", "encode_png"]

_ACCENT = (255, 176, 32)  # crop rectangle, BGR
_MEASURED = (120, 255, 120)  # measured landmarks
_ESTIMATED = (140, 170, 255)  # inferred crown and chin
_FACE_BOX = (200, 200, 200)
_PREVIEW_MAX_EDGE = 900


def encode_png(image_rgb: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG data URL."""
    ok, buffer = cv2.imencode(".png", image_rgb[:, :, ::-1])
    if not ok:  # pragma: no cover - cv2 only fails here on malformed input
        msg = "could not encode preview image"
        raise PortraitKitError(msg)
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


@dataclass(frozen=True, slots=True)
class InspectorOptions:
    """Choices the UI offers, derived from the registries rather than hard-coded."""

    detectors: tuple[dict[str, Any], ...]
    presets: tuple[dict[str, Any], ...]
    default_detector: str
    default_preset: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detectors": list(self.detectors),
            "presets": list(self.presets),
            "default_detector": self.default_detector,
            "default_preset": self.default_preset,
        }


def _fit_preview(image: np.ndarray, max_edge: int = _PREVIEW_MAX_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


def _dashed_line(
    canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], colour, dash: int = 9
) -> None:
    """Draw a dashed line, used for quantities that are inferred rather than measured."""
    (x1, y1), (x2, y2) = start, end
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for offset in range(0, length, dash * 2):
        a = offset / length
        b = min((offset + dash) / length, 1.0)
        cv2.line(
            canvas,
            (round(x1 + (x2 - x1) * a), round(y1 + (y2 - y1) * a)),
            (round(x1 + (x2 - x1) * b), round(y1 + (y2 - y1) * b)),
            colour,
            1,
            cv2.LINE_AA,
        )


def render_overlay(loaded: LoadedImage, detection: DetectionResult, plan: Any | None) -> np.ndarray:
    """Draw what the pipeline measured on top of the source image.

    Solid marks are measured directly from landmarks; dashed marks are inferred. Encoding
    that difference visually is the point of the overlay -- the compliance panel says
    which checks rest on estimates, and the picture should agree with it.
    """
    canvas = loaded.pixels.copy()[:, :, ::-1].copy()  # work in BGR for cv2
    primary = detection.primary

    if primary is not None:
        box = primary.box
        cv2.rectangle(
            canvas,
            (round(box.x1), round(box.y1)),
            (round(box.x2), round(box.y2)),
            _FACE_BOX,
            1,
            cv2.LINE_AA,
        )
        landmarks = primary.landmarks
        if landmarks is not None:
            for point in landmarks.as_points():
                cv2.circle(canvas, (round(point.x), round(point.y)), 3, _MEASURED, -1, cv2.LINE_AA)
            left, right = landmarks.left_eye, landmarks.right_eye
            cv2.line(
                canvas,
                (round(left.x), round(left.y)),
                (round(right.x), round(right.y)),
                _MEASURED,
                1,
                cv2.LINE_AA,
            )
            centre = landmarks.eye_center
            cv2.drawMarker(
                canvas,
                (round(centre.x), round(centre.y)),
                _MEASURED,
                cv2.MARKER_CROSS,
                14,
                1,
                cv2.LINE_AA,
            )

    if plan is not None:
        rect = plan.rect
        cv2.rectangle(
            canvas,
            (round(rect.x1), round(rect.y1)),
            (round(rect.x2), round(rect.y2)),
            _ACCENT,
            2,
            cv2.LINE_AA,
        )
        head = plan.head
        for y in (head.crown.y, head.chin.y):
            _dashed_line(canvas, (round(rect.x1), round(y)), (round(rect.x2), round(y)), _ESTIMATED)

    return _fit_preview(canvas[:, :, ::-1])


class AnalysisService:
    """Runs the pipeline for the inspector and shapes the result for display.

    Detectors are built once per name and reused. Constructing an ONNX session costs far
    more than an inference, so a fresh one per request would make the UI feel slow for a
    reason that has nothing to do with the pipeline.
    """

    def __init__(self, *, allow_download: bool | None = None) -> None:
        self._allow_download = allow_download
        self._detectors: dict[str, DetectionStage] = {}

    def options(self) -> InspectorOptions:
        """Describe the detectors and presets the UI may offer."""
        return InspectorOptions(
            detectors=tuple(
                {
                    "name": name,
                    "license": spec.license,
                    "commercial": spec.permits_commercial_use,
                    "size_mib": round(spec.size_bytes / (1024 * 1024), 2),
                }
                for name, spec in MODELS.items()
            ),
            presets=tuple(
                {
                    "name": name,
                    "description": preset.description,
                    "width": preset.output_size.width,
                    "height": preset.output_size.height,
                    "claims_compliance": preset.makes_compliance_claim,
                }
                for name, preset in PRESETS.items()
            ),
            default_detector="yunet-2023mar",
            default_preset=DEFAULT_PRESET,
        )

    def samples(self) -> tuple[Path, ...]:
        """Public sample portraits, if the manifest-fetched set is present.

        Bundling a one-click sample matters more than it looks: a reviewer opening this
        for the first time rarely has a passport photo to hand, and a demo that needs one
        before it shows anything is a demo nobody sees.
        """
        root = load_settings().data_dir / "public-samples"
        if not root.is_dir():
            return ()
        return tuple(sorted(path for path in root.rglob("*.jpg") if path.is_file()))

    def sample_bytes(self, index: int) -> bytes | None:
        """Return one sample's bytes, or ``None`` when the index is out of range."""
        available = self.samples()
        if not 0 <= index < len(available):
            return None
        return available[index].read_bytes()

    def _stage_for(self, detector: str) -> DetectionStage:
        if detector not in self._detectors:
            self._detectors[detector] = DetectionStage(
                build_detector(detector, allow_download=self._allow_download)
            )
        return self._detectors[detector]

    def analyse(
        self, image_bytes: bytes, *, detector: str = "yunet-2023mar", preset: str = DEFAULT_PRESET
    ) -> dict[str, Any]:
        """Run detection and cropping over ``image_bytes`` and return display data.

        Failures are returned as data with an ``error`` key rather than raised, because
        the UI needs to render them and an unhandled exception would show a blank panel.
        """
        started = time.perf_counter()
        try:
            loaded = load_image(image_bytes)
            stage = self._stage_for(detector)
            detection = stage.run(loaded)
            crop = CropStage(CropConfig(preset=preset)).run(loaded, detection)
        except PortraitKitError as error:
            return {"ok": False, "error": str(error)}

        plan = crop.plan
        primary = detection.primary
        payload: dict[str, Any] = {
            "ok": crop.image is not None,
            "detector": detector,
            "preset": preset,
            "source": {
                "width": loaded.size.width,
                "height": loaded.size.height,
                "orientation_corrected": loaded.orientation.applied,
                "truncated": loaded.truncated,
            },
            "detection": {
                "status": str(detection.status),
                "faces": detection.face_count,
                "score": None if primary is None else round(primary.score, 4),
                "roll_degrees": (
                    None
                    if primary is None or primary.landmarks is None
                    else round(primary.landmarks.roll_degrees, 2)
                ),
                "duration_ms": round(detection.duration_ms, 1),
                "diagnostics": [str(item) for item in detection.diagnostics],
            },
            "crop": {
                "status": str(crop.status),
                "conforms": crop.conforms,
                "padded": crop.padded,
                "duration_ms": round(crop.duration_ms, 1),
                "derotated_degrees": (
                    None if crop.derotation is None else round(crop.derotation.angle_degrees, 2)
                ),
            },
            "overlay": encode_png(render_overlay(loaded, detection, plan)),
            "output": None if crop.image is None else encode_png(crop.image),
            "checks": self._checks(crop),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        return payload

    @staticmethod
    def _checks(crop: Any) -> list[dict[str, Any]]:
        if crop.assessment is None:
            return []
        rows = []
        for check in crop.assessment.checks:
            permitted = check.permitted
            rows.append(
                {
                    "name": check.name.replace("_", " "),
                    "status": str(check.status),
                    "basis": str(check.basis),
                    "value": None if check.value is None else round(check.value, 4),
                    "low": None if permitted is None else round(permitted[0], 4),
                    "high": (
                        None
                        if permitted is None or permitted[1] == float("inf")
                        else round(permitted[1], 4)
                    ),
                    "clause": check.clause,
                    "detail": check.detail,
                    "passed": check.status is CheckStatus.PASS,
                    "estimated": check.basis is CheckBasis.ESTIMATED,
                }
            )
        return rows
