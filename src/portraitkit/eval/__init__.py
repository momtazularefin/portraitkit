"""Evaluation twins for PortraitKit pipeline stages."""

from portraitkit.eval.annotations import (
    AnnotatedFace,
    AnnotatedImage,
    AnnotationSet,
    load_annotations,
)
from portraitkit.eval.crop import CropQualityAggregate, CropQualityReport, evaluate_crop_quality
from portraitkit.eval.matching import Match, MatchResult, match_detections
from portraitkit.eval.matting import (
    MattingEvaluation,
    MattingEvaluationReport,
    MattingEvaluationSummary,
    evaluate_matting,
)
from portraitkit.eval.matting_annotations import (
    MattingAnnotationSet,
    MattingSample,
    load_matting_annotations,
)
from portraitkit.eval.metrics import DetectionCounts, normalized_landmark_error
from portraitkit.eval.report import EvaluationReport, ImageEvaluation
from portraitkit.eval.runner import evaluate_detection
from portraitkit.eval.samples import (
    PublicSampleManifest,
    PublicSampleSpec,
    ResolvedPublicSamples,
    load_public_sample_manifest,
    resolve_public_samples,
)

__all__ = [
    "AnnotatedFace",
    "AnnotatedImage",
    "AnnotationSet",
    "CropQualityAggregate",
    "CropQualityReport",
    "DetectionCounts",
    "EvaluationReport",
    "ImageEvaluation",
    "Match",
    "MatchResult",
    "MattingAnnotationSet",
    "MattingEvaluation",
    "MattingEvaluationReport",
    "MattingEvaluationSummary",
    "MattingSample",
    "PublicSampleManifest",
    "PublicSampleSpec",
    "ResolvedPublicSamples",
    "evaluate_crop_quality",
    "evaluate_detection",
    "evaluate_matting",
    "load_annotations",
    "load_matting_annotations",
    "load_public_sample_manifest",
    "match_detections",
    "normalized_landmark_error",
    "resolve_public_samples",
]
