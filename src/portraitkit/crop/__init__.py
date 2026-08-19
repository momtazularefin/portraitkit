"""Stage 2: ICAO-style geometry cropping and its conformance assessment."""

from portraitkit.crop.compliance import (
    CheckBasis,
    CheckStatus,
    GeometryAssessment,
    GeometryCheck,
    assess_geometry,
)
from portraitkit.crop.derotate import Derotation, level_eye_line
from portraitkit.crop.geometry import CropPlan, HeadEstimate, estimate_head, solve_crop
from portraitkit.crop.ofiq import (
    CROP_QUALITY_MEASURES,
    OFIQ_REFERENCE,
    OfiqComparison,
    OfiqInstallation,
    OfiqMeasurement,
    OfiqProvenance,
    OfiqResult,
    OfiqScorer,
    resolve_reference_ofiq,
)
from portraitkit.crop.presets import DEFAULT_PRESET, PRESETS, CropPreset, get_preset, preset_names
from portraitkit.crop.stage import CropConfig, CropResult, CropStage, CropStatus

__all__ = [
    "CROP_QUALITY_MEASURES",
    "DEFAULT_PRESET",
    "OFIQ_REFERENCE",
    "PRESETS",
    "CheckBasis",
    "CheckStatus",
    "CropConfig",
    "CropPlan",
    "CropPreset",
    "CropResult",
    "CropStage",
    "CropStatus",
    "Derotation",
    "GeometryAssessment",
    "GeometryCheck",
    "HeadEstimate",
    "OfiqComparison",
    "OfiqInstallation",
    "OfiqMeasurement",
    "OfiqProvenance",
    "OfiqResult",
    "OfiqScorer",
    "assess_geometry",
    "estimate_head",
    "get_preset",
    "level_eye_line",
    "preset_names",
    "resolve_reference_ofiq",
    "solve_crop",
]
