"""Unit tests for matting evaluation runner and annotation manifests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from portraitkit.errors import AnnotationError
from portraitkit.eval.matting import evaluate_matting
from portraitkit.eval.matting_annotations import (
    load_matting_annotations,
)
from portraitkit.matting.stage import MattingStage
from tests.test_matting_stage import DummyMatter


def test_load_matting_annotations_parses_manifest(tmp_path: Path) -> None:
    manifest_data = {
        "schema_version": 1,
        "name": "test-dataset",
        "samples": [
            {
                "image": "img1.jpg",
                "alpha": "mask1.png",
                "trimap": "trimap1.png",
            },
            {
                "image": "img2.jpg",
                "alpha": "mask2.png",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    dataset = load_matting_annotations(manifest_path)

    assert dataset.name == "test-dataset"
    assert len(dataset.samples) == 2
    assert dataset.samples[0].relative_image == "img1.jpg"
    assert dataset.samples[0].relative_alpha == "mask1.png"
    assert dataset.samples[0].relative_trimap == "trimap1.png"
    assert dataset.samples[1].trimap_path is None


def test_load_matting_annotations_rejects_invalid_version(tmp_path: Path) -> None:
    manifest_data = {"schema_version": 99, "samples": []}
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(AnnotationError, match="unsupported schema_version"):
        load_matting_annotations(manifest_path)


def test_load_matting_annotations_rejects_missing_file() -> None:
    with pytest.raises(AnnotationError, match="does not exist"):
        load_matting_annotations("nonexistent.json")


def test_evaluate_matting_computes_metrics_and_summary(tmp_path: Path) -> None:
    # Create synthetic image and ground-truth alpha
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    alpha = np.zeros((50, 50), dtype=np.uint8)
    alpha[:, :25] = 255  # Left half is 1.0

    img_path = tmp_path / "test.jpg"
    alpha_path = tmp_path / "test_alpha.png"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(alpha_path), alpha)

    manifest_data = {
        "schema_version": 1,
        "name": "synth-eval",
        "samples": [
            {
                "image": "test.jpg",
                "alpha": "test_alpha.png",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Dummy matter predicts left half 1.0, right half 0.0 -> perfect match with alpha!
    matter = DummyMatter()
    stage = MattingStage(matter)
    annotations = load_matting_annotations(manifest_path)

    report = evaluate_matting(stage, annotations)

    assert report.dataset == "synth-eval"
    assert report.matter == "dummy-matter"
    assert report.summary.evaluated_samples == 1
    assert report.summary.errored_samples == 0
    assert report.summary.mean_sad == pytest.approx(0.0)
    assert report.summary.mean_mse == pytest.approx(0.0)
    assert len(report.samples) == 1
    assert report.samples[0].metrics.sad == pytest.approx(0.0)

    # Verify write() and JSON serialization
    report_file = tmp_path / "report.json"
    report.write(report_file)
    assert report_file.exists()
    loaded_json = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded_json["dataset"] == "synth-eval"
