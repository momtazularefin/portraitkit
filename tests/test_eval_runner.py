"""Annotation loading, the evaluation runner, and report serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from portraitkit.detection.selection import SelectionStrategy
from portraitkit.detection.stage import DetectionStage, StageConfig
from portraitkit.errors import AnnotationError
from portraitkit.eval.annotations import load_annotations
from portraitkit.eval.metrics import DetectionCounts
from portraitkit.eval.report import EvaluationReport, ImageEvaluation
from portraitkit.eval.runner import evaluate_detection
from portraitkit.types import BoundingBox, FaceDetection, FaceLandmarks5
from tests.conftest import solid_image
from tests.test_detection_stage import StubDetector

BOX_A = [20.0, 20.0, 80.0, 100.0]
BOX_B = [120.0, 30.0, 170.0, 100.0]
POINTS_A = [[35.0, 45.0], [65.0, 45.0], [50.0, 65.0], [38.0, 85.0], [62.0, 85.0]]


def write_manifest(tmp_path: Path, images: list[dict], name: str = "example-set") -> Path:
    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "name": name, "images": images}), encoding="utf-8"
    )
    return manifest


def write_image(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(solid_image(200, 200), mode="RGB").save(path)


def detection(values: list[float], score: float = 0.9, points: list | None = None) -> FaceDetection:
    landmarks = (
        None if points is None else FaceLandmarks5.from_array(np.asarray(points, dtype=np.float32))
    )
    return FaceDetection(box=BoundingBox(*values), score=score, landmarks=landmarks)


# --- annotations ---------------------------------------------------------------------


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A, "landmarks": POINTS_A}]}]
    )

    annotations = load_annotations(manifest)

    assert annotations.name == "example-set"
    assert len(annotations) == 1
    assert annotations.face_count == 1
    face = annotations.images[0].faces[0]
    assert face.box.as_tuple() == (20.0, 20.0, 80.0, 100.0)
    assert face.landmarks is not None
    assert face.landmarks.interocular_distance == pytest.approx(30.0)


def test_image_paths_resolve_against_the_manifest_directory(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [{"path": "portraits/a.jpg", "faces": []}])

    annotations = load_annotations(manifest)

    assert annotations.images[0].path == tmp_path / "portraits" / "a.jpg"


def test_an_explicit_root_overrides_the_manifest_directory(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": []}])

    annotations = load_annotations(manifest, root=tmp_path / "elsewhere")

    assert annotations.images[0].path == tmp_path / "elsewhere" / "a.jpg"


def test_a_manifest_validates_without_the_images_present(tmp_path: Path) -> None:
    """Manifests travel; images do not. Validation must not require the pixels."""
    manifest = write_manifest(tmp_path, [{"path": "absent.jpg", "faces": [{"box": BOX_A}]}])

    assert load_annotations(manifest).face_count == 1


def test_primary_flag_is_read(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [{"path": "a.jpg", "faces": [{"box": BOX_A}, {"box": BOX_B, "primary": True}]}],
    )

    annotations = load_annotations(manifest)

    assert annotations.images_with_primary == 1
    primary = annotations.images[0].primary
    assert primary is not None
    assert primary.box.x1 == pytest.approx(120.0)


def test_two_primaries_are_rejected(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [
            {
                "path": "a.jpg",
                "faces": [{"box": BOX_A, "primary": True}, {"box": BOX_B, "primary": True}],
            }
        ],
    )

    with pytest.raises(AnnotationError, match="at most one face may be marked primary"):
        load_annotations(manifest)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="cannot read annotation manifest"):
        load_annotations(tmp_path / "absent.json")


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "annotations.json"
    manifest.write_text("{not json", encoding="utf-8")

    with pytest.raises(AnnotationError, match="not valid JSON"):
        load_annotations(manifest)


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "annotations.json"
    manifest.write_text(json.dumps({"schema_version": 99, "images": []}), encoding="utf-8")

    with pytest.raises(AnnotationError, match="unsupported schema_version"):
        load_annotations(manifest)


@pytest.mark.parametrize(
    ("faces", "expected"),
    [
        ([{"box": [1, 2, 3]}], "four numbers"),
        ([{"box": [10, 10, 5, 20]}], "x2 >= x1"),
        ([{"box": BOX_A, "landmarks": [[1, 2]]}], "five"),
        ([{"nose": 1}], "missing its box"),
    ],
)
def test_malformed_faces_name_the_problem(tmp_path: Path, faces: list, expected: str) -> None:
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": faces}])

    with pytest.raises(AnnotationError, match=expected):
        load_annotations(manifest)


def test_error_message_identifies_the_offending_image(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [{"path": "good.jpg", "faces": []}, {"path": "bad.jpg", "faces": [{"box": [1, 2, 3]}]}],
    )

    with pytest.raises(AnnotationError, match=r"image 1 \(bad\.jpg\)"):
        load_annotations(manifest)


# --- runner --------------------------------------------------------------------------


def test_perfect_detector_scores_perfectly(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A}]}])
    stage = DetectionStage(StubDetector((detection(BOX_A),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.counts == DetectionCounts(true_positives=1)
    assert report.to_dict()["summary"]["f1"] == pytest.approx(1.0)
    assert report.mean_iou == pytest.approx(1.0)


def test_missed_and_spurious_faces_are_counted(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A}]}])
    stage = DetectionStage(StubDetector((detection([150.0, 150.0, 190.0, 190.0]),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.counts == DetectionCounts(false_positives=1, false_negatives=1)
    assert report.mean_iou is None


def test_counts_accumulate_across_images(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    write_image(tmp_path, "b.jpg")
    manifest = write_manifest(
        tmp_path,
        [
            {"path": "a.jpg", "faces": [{"box": BOX_A}]},
            {"path": "b.jpg", "faces": [{"box": BOX_A}]},
        ],
    )
    stage = DetectionStage(StubDetector((detection(BOX_A),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.counts.true_positives == 2
    assert report.evaluated_images == 2


def test_landmark_error_is_reported_when_both_sides_have_points(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(
        tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A, "landmarks": POINTS_A}]}]
    )
    shifted = [[x + 3.0, y] for x, y in POINTS_A]
    stage = DetectionStage(StubDetector((detection(BOX_A, points=shifted),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.mean_landmark_error == pytest.approx(3.0 / 30.0)


def test_landmark_error_is_absent_when_ground_truth_has_none(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A}]}])
    stage = DetectionStage(StubDetector((detection(BOX_A, points=POINTS_A),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.mean_landmark_error is None


def test_primary_selection_is_scored_correct(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(
        tmp_path,
        [{"path": "a.jpg", "faces": [{"box": BOX_B}, {"box": BOX_A, "primary": True}]}],
    )
    # BOX_A is the larger face, so the default largest-first strategy should pick it.
    stage = DetectionStage(StubDetector((detection(BOX_B, score=0.95), detection(BOX_A))))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.primary_accuracy == pytest.approx(1.0)


def test_primary_selection_is_scored_wrong_when_the_strategy_misses(tmp_path: Path) -> None:
    """Selection is measured, not assumed. A strategy that picks the wrong face says so."""
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(
        tmp_path,
        [{"path": "a.jpg", "faces": [{"box": BOX_B, "primary": True}, {"box": BOX_A}]}],
    )
    stage = DetectionStage(
        StubDetector((detection(BOX_B, score=0.95), detection(BOX_A))),
        StageConfig(selection=SelectionStrategy.LARGEST),
    )

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.primary_accuracy == pytest.approx(0.0)


def test_primary_accuracy_is_none_when_nothing_declares_a_primary(tmp_path: Path) -> None:
    """An unmeasured quantity must not read as a perfect score."""
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A}]}])
    stage = DetectionStage(StubDetector((detection(BOX_A),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.primary_accuracy is None


def test_unloadable_images_are_recorded_not_skipped(tmp_path: Path) -> None:
    """A run that silently dropped half its inputs would look like a clean sweep."""
    write_image(tmp_path, "good.jpg")
    manifest = write_manifest(
        tmp_path,
        [
            {"path": "good.jpg", "faces": [{"box": BOX_A}]},
            {"path": "absent.jpg", "faces": [{"box": BOX_A}]},
        ],
    )
    stage = DetectionStage(StubDetector((detection(BOX_A),)))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert report.evaluated_images == 1
    assert len(report.errors) == 1
    assert report.errors[0][0] == "absent.jpg"
    assert report.to_dict()["summary"]["errored_images"] == 1


def test_report_records_the_settings_needed_to_reproduce_it(tmp_path: Path) -> None:
    write_image(tmp_path, "a.jpg")
    manifest = write_manifest(tmp_path, [{"path": "a.jpg", "faces": [{"box": BOX_A}]}])
    stage = DetectionStage(StubDetector((detection(BOX_A),), name="stub-model"))

    report = evaluate_detection(stage, load_annotations(manifest), iou_threshold=0.7)

    assert report.detector == "stub-model"
    assert report.dataset == "example-set"
    assert report.settings["iou_threshold"] == pytest.approx(0.7)
    assert report.settings["selection"] == "largest"


# --- report serialization ------------------------------------------------------------


def sample_report() -> EvaluationReport:
    return EvaluationReport(
        dataset="set",
        detector="stub",
        settings={"iou_threshold": 0.5},
        images=(
            ImageEvaluation(
                path="a.jpg",
                counts=DetectionCounts(true_positives=1),
                match_ious=(1 / 3,),
                duration_ms=1 / 7,
            ),
        ),
        counts=DetectionCounts(true_positives=1),
    )


def test_serialization_is_byte_identical_across_runs() -> None:
    assert sample_report().to_json() == sample_report().to_json()


def test_floats_are_rounded_for_stable_diffs() -> None:
    payload = json.loads(sample_report().to_json())

    assert payload["images"][0]["mean_iou"] == pytest.approx(0.333333)
    assert payload["images"][0]["duration_ms"] == pytest.approx(0.142857)


def test_images_keep_manifest_order(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.jpg"):
        write_image(tmp_path, name)
    manifest = write_manifest(
        tmp_path,
        [{"path": "b.jpg", "faces": []}, {"path": "a.jpg", "faces": []}],
    )
    stage = DetectionStage(StubDetector(()))

    report = evaluate_detection(stage, load_annotations(manifest))

    assert [image["path"] for image in report.to_dict()["images"]] == ["b.jpg", "a.jpg"]


def test_report_writes_to_disk(tmp_path: Path) -> None:
    target = sample_report().write(tmp_path / "nested" / "report.json")

    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["detector"] == "stub"
