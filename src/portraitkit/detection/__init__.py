"""Stage 1: face detection, landmarks, and primary-subject selection."""

from portraitkit.detection.base import DetectorConfig, FaceDetector, RawDetections
from portraitkit.detection.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.detection.nms import non_max_suppression
from portraitkit.detection.scrfd import ScrfdDetector, decode_scrfd
from portraitkit.detection.selection import SelectionStrategy, select_primary
from portraitkit.detection.stage import DetectionStage, StageConfig, build_detector
from portraitkit.detection.yunet import YuNetDetector, decode_yunet

__all__ = [
    "ColorOrder",
    "DetectionStage",
    "DetectorConfig",
    "FaceDetector",
    "PreprocessContract",
    "RawDetections",
    "ResizeMode",
    "ScrfdDetector",
    "SelectionStrategy",
    "StageConfig",
    "TensorLayout",
    "YuNetDetector",
    "build_detector",
    "decode_scrfd",
    "decode_yunet",
    "non_max_suppression",
    "select_primary",
]
