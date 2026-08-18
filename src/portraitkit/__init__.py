"""PortraitKit: portrait processing pipeline with stage-by-stage evaluation."""

from portraitkit.errors import (
    ConfigError,
    ImageLoadError,
    ModelError,
    ModelIntegrityError,
    ModelNotAvailableError,
    PortraitKitError,
    StageError,
)
from portraitkit.types import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    Diagnostic,
    FaceDetection,
    FaceLandmarks5,
    ImageSize,
    Point,
)

__version__ = "0.1.0"

__all__ = [
    "BoundingBox",
    "ConfigError",
    "DetectionResult",
    "DetectionStatus",
    "Diagnostic",
    "FaceDetection",
    "FaceLandmarks5",
    "ImageLoadError",
    "ImageSize",
    "ModelError",
    "ModelIntegrityError",
    "ModelNotAvailableError",
    "Point",
    "PortraitKitError",
    "StageError",
    "__version__",
]
