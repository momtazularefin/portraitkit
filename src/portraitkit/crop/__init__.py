"""Stage 2: ICAO-style geometry cropping and its conformance assessment."""

from portraitkit.crop.compliance import (
    CheckBasis,
    CheckStatus,
    GeometryAssessment,
    GeometryCheck,
    assess_geometry,
)
from portraitkit.crop.geometry import CropPlan, HeadEstimate, estimate_head, solve_crop
from portraitkit.crop.presets import DEFAULT_PRESET, PRESETS, CropPreset, get_preset, preset_names
from portraitkit.crop.stage import CropConfig, CropResult, CropStage, CropStatus

__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "CheckBasis",
    "CheckStatus",
    "CropConfig",
    "CropPlan",
    "CropPreset",
    "CropResult",
    "CropStage",
    "CropStatus",
    "GeometryAssessment",
    "GeometryCheck",
    "HeadEstimate",
    "assess_geometry",
    "estimate_head",
    "get_preset",
    "preset_names",
    "solve_crop",
]
