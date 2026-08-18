"""Running the detection stage over an annotation set.

The runner is deliberately thin: load, detect, match, count. Everything it reports is
derived from the same stage a caller would use in production, so the evaluation measures
the shipped path rather than a parallel one built to look good.
"""

from __future__ import annotations

from portraitkit.detection.stage import DetectionStage
from portraitkit.errors import ImageLoadError
from portraitkit.eval.annotations import AnnotatedImage, AnnotationSet
from portraitkit.eval.matching import MatchResult, match_detections
from portraitkit.eval.metrics import DetectionCounts, normalized_landmark_error
from portraitkit.eval.report import EvaluationReport, ImageEvaluation
from portraitkit.imaging.io import load_image
from portraitkit.types import DetectionResult

__all__ = ["evaluate_detection"]


def evaluate_detection(
    stage: DetectionStage,
    annotations: AnnotationSet,
    *,
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Evaluate ``stage`` against ``annotations``.

    Args:
        stage: The detection stage to measure.
        annotations: Ground truth to measure against.
        iou_threshold: Overlap required for a prediction to count as a match.

    Returns:
        A report carrying per-image outcomes, aggregate counts, and the settings needed
        to reproduce the run. Images that fail to load are recorded as errors rather than
        silently skipped.
    """
    detector = stage.detector
    evaluations: list[ImageEvaluation] = []
    errors: list[tuple[str, str]] = []
    totals = DetectionCounts()

    for annotated in annotations.images:
        try:
            image = load_image(annotated.path)
        except ImageLoadError as error:
            errors.append((annotated.relative_path, str(error)))
            continue

        result = stage.run(image)
        truth_boxes = [face.box for face in annotated.faces]
        matched = match_detections(truth_boxes, result.faces, iou_threshold=iou_threshold)

        counts = DetectionCounts(
            true_positives=matched.true_positives,
            false_positives=matched.false_positives,
            false_negatives=matched.false_negatives,
        )
        totals = totals + counts

        landmark_errors: list[float] = []
        for match in matched.matches:
            expected = annotated.faces[match.truth_index].landmarks
            actual = result.faces[match.prediction_index].landmarks
            if expected is None or actual is None:
                continue
            try:
                landmark_errors.append(normalized_landmark_error(expected, actual))
            except ValueError:
                # Degenerate ground truth with coincident eyes cannot be normalized;
                # excluding it is honest, silently scoring it zero would not be.
                continue

        evaluations.append(
            ImageEvaluation(
                path=annotated.relative_path,
                counts=counts,
                match_ious=tuple(match.iou for match in matched.matches),
                landmark_errors=tuple(landmark_errors),
                primary_correct=_judge_primary(annotated, result, matched),
                duration_ms=result.duration_ms,
                diagnostics=tuple(str(item) for item in result.diagnostics),
            )
        )

    return EvaluationReport(
        dataset=annotations.name,
        detector=detector.name,
        settings={
            "iou_threshold": iou_threshold,
            "score_threshold": detector.config.score_threshold,
            "nms_iou_threshold": detector.config.nms_iou_threshold,
            "max_faces": detector.config.max_faces,
            "selection": str(stage.config.selection),
            "providers": list(detector.info.providers),
        },
        images=tuple(evaluations),
        errors=tuple(errors),
        counts=totals,
    )


def _judge_primary(
    annotated: AnnotatedImage, result: DetectionResult, matched: MatchResult
) -> bool | None:
    """Whether the stage selected the annotated primary subject.

    Returns ``None`` when the question does not apply: the annotation declares no
    primary, or the stage found no face to select.
    """
    expected = annotated.primary
    if expected is None or result.primary is None:
        return None

    truth_index = annotated.faces.index(expected)
    prediction_index = matched.prediction_for(truth_index)
    if prediction_index is None:
        # The annotated subject was never detected, so the selection cannot be right.
        return False
    return result.faces[prediction_index] is result.primary
