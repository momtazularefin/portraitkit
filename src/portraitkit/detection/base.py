"""The face-detector adapter interface.

Every detector PortraitKit supports, and every detector PortraitBench grades, is reached
through this interface. Adapters differ only in their declared preprocessing contract and
their decoding of raw model output; everything downstream of that -- suppression, score
filtering, clipping, ordering -- is shared, so a benchmark comparison reflects the models
rather than their wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from portraitkit.detection.contract import PreprocessContract
from portraitkit.detection.nms import non_max_suppression
from portraitkit.models.session import CPU_PROVIDER, create_session, describe_session
from portraitkit.types import BoundingBox, FaceDetection, FaceLandmarks5, ImageSize

if TYPE_CHECKING:
    from portraitkit.imaging.geometry import ResizeTransform

__all__ = ["DetectorConfig", "FaceDetector", "RawDetections"]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Thresholds shared by every detector adapter."""

    score_threshold: float = 0.6
    """Minimum confidence for a candidate to be reported."""

    nms_iou_threshold: float = 0.3
    """Overlap above which a lower-scoring duplicate is suppressed."""

    top_k: int = 500
    """Candidates considered during suppression, highest score first."""

    max_faces: int = 20
    """Upper bound on reported faces."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            msg = f"score_threshold must be in [0, 1], got {self.score_threshold}"
            raise ValueError(msg)
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            msg = f"nms_iou_threshold must be in [0, 1], got {self.nms_iou_threshold}"
            raise ValueError(msg)
        if self.top_k < 1:
            msg = f"top_k must be at least 1, got {self.top_k}"
            raise ValueError(msg)
        if self.max_faces < 1:
            msg = f"max_faces must be at least 1, got {self.max_faces}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RawDetections:
    """Decoded model output in model-input coordinates, before shared postprocessing.

    Attributes:
        boxes: ``(N, 4)`` array of ``x1, y1, x2, y2``.
        scores: ``(N,)`` array of confidences.
        landmarks: ``(N, 5, 2)`` array, or ``None`` for a detector without landmarks.
    """

    boxes: np.ndarray
    scores: np.ndarray
    landmarks: np.ndarray | None = None


class FaceDetector(ABC):
    """Base class for ONNX face-detector adapters.

    Subclasses declare a preprocessing contract and implement :meth:`decode`. The
    template method :meth:`detect` owns everything else.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        config: DetectorConfig | None = None,
        providers: tuple[str, ...] = (CPU_PROVIDER,),
    ) -> None:
        self.config = config or DetectorConfig()
        self.session = create_session(model_path, providers=providers)
        self.info = describe_session(self.session)
        # Fail at construction if the artifact contradicts the declared contract, rather
        # than producing plausible-looking nonsense at inference time.
        self.contract.validate_against(self.info)

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name of the model this adapter drives."""

    @property
    @abstractmethod
    def contract(self) -> PreprocessContract:
        """The preprocessing this adapter declares."""

    @abstractmethod
    def decode(self, outputs: list[np.ndarray]) -> RawDetections:
        """Turn raw session output into candidates in model-input coordinates."""

    def detect(self, image_rgb: np.ndarray) -> tuple[FaceDetection, ...]:
        """Detect faces in an upright RGB image.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 array in RGB order.

        Returns:
            Detections in source-image coordinates, highest confidence first.
        """
        tensor, transform = self.contract.build_input(image_rgb)
        outputs = self.session.run(None, {self.contract.input_name: tensor})
        raw = self.decode(list(outputs))
        return self._postprocess(raw, transform)

    def _postprocess(
        self, raw: RawDetections, transform: ResizeTransform
    ) -> tuple[FaceDetection, ...]:
        """Filter, suppress, map back to source coordinates, and clip."""
        keep = raw.scores >= self.config.score_threshold
        boxes = raw.boxes[keep]
        scores = raw.scores[keep]
        landmarks = None if raw.landmarks is None else raw.landmarks[keep]
        if boxes.size == 0:
            return ()

        selected = non_max_suppression(
            boxes,
            scores,
            iou_threshold=self.config.nms_iou_threshold,
            top_k=self.config.top_k,
        )[: self.config.max_faces]

        source_size = ImageSize(width=transform.source.width, height=transform.source.height)
        detections: list[FaceDetection] = []
        for index in selected:
            box = transform.invert_box(
                BoundingBox(*(float(value) for value in boxes[index]))
            ).clipped_to(source_size)
            if box.area <= 0.0:
                # Entirely outside the frame after inversion; nothing usable remains.
                continue
            points = (
                None
                if landmarks is None
                else FaceLandmarks5.from_array(transform.invert_array(landmarks[index]))
            )
            detections.append(
                FaceDetection(
                    box=box,
                    score=float(min(max(scores[index], 0.0), 1.0)),
                    landmarks=points,
                )
            )
        return tuple(detections)
