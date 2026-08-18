"""Evaluation twin for the detection stage."""

from portraitkit.eval.annotations import (
    AnnotatedFace,
    AnnotatedImage,
    AnnotationSet,
    load_annotations,
)
from portraitkit.eval.matching import Match, MatchResult, match_detections
from portraitkit.eval.metrics import DetectionCounts, normalized_landmark_error
from portraitkit.eval.report import EvaluationReport, ImageEvaluation
from portraitkit.eval.runner import evaluate_detection

__all__ = [
    "AnnotatedFace",
    "AnnotatedImage",
    "AnnotationSet",
    "DetectionCounts",
    "EvaluationReport",
    "ImageEvaluation",
    "Match",
    "MatchResult",
    "evaluate_detection",
    "load_annotations",
    "match_detections",
    "normalized_landmark_error",
]
