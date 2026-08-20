"""Unit tests for the PortraitBench execution runner, config parsing, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from portraitkit.bench.cli import EXIT_OK
from portraitkit.bench.cli import main as bench_main
from portraitkit.bench.config import (
    BenchmarkConfig,
    DatasetSpec,
    DegradationSpec,
    load_benchmark_config,
)
from portraitkit.bench.runner import BenchmarkRunner
from portraitkit.types import BoundingBox, FaceLandmarks5, Point
from tests.conftest import StubDetector


def test_load_benchmark_config_parses_json(tmp_path: Path) -> None:
    config_dict = {
        "schema_version": 1,
        "name": "test-suite",
        "models": ["fake-detector"],
        "datasets": [
            {
                "name": "test-data",
                "track": "detection",
                "manifest": "manifest.json",
            }
        ],
        "degradations": [
            {"name": "clean", "steps": []},
            {"name": "jpeg_heavy", "steps": [{"type": "jpeg", "quality": 20}]},
        ],
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config_dict), encoding="utf-8")

    cfg = load_benchmark_config(cfg_file)

    assert cfg.name == "test-suite"
    assert cfg.models == ("fake-detector",)
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].name == "test-data"
    assert len(cfg.degradations) == 2
    assert cfg.degradations[1].name == "jpeg_heavy"


def test_benchmark_runner_executes_detection_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Create a test image and annotation manifest
    img_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (100, 100), color="white").save(img_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "schema_version": 1,
        "name": "bench-det-dataset",
        "images": [
            {
                "path": "portrait.jpg",
                "faces": [
                    {
                        "box": [20, 20, 80, 80],
                        "landmarks": [[35, 40], [65, 40], [50, 60], [40, 75], [60, 75]],
                    }
                ],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # 2. Mock build_detector
    fake_detector = StubDetector(
        boxes=[BoundingBox(20.0, 20.0, 80.0, 80.0)],
        landmarks=[
            FaceLandmarks5(
                left_eye=Point(35.0, 40.0),
                right_eye=Point(65.0, 40.0),
                nose=Point(50.0, 60.0),
                left_mouth=Point(40.0, 75.0),
                right_mouth=Point(60.0, 75.0),
            )
        ],
        name="fake-detector",
    )
    import portraitkit.bench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "build_detector", lambda *args, **kwargs: fake_detector)

    # 3. Create benchmark config
    config = BenchmarkConfig(
        name="test-run",
        models=("fake-detector",),
        datasets=(
            DatasetSpec(
                name="bench-det-dataset",
                track="detection",
                manifest_path=str(manifest_path),
                root_path=str(tmp_path),
            ),
        ),
        degradations=(
            DegradationSpec(name="clean", steps=()),
            DegradationSpec(name="jpeg_heavy", steps=({"type": "jpeg", "quality": 20},)),
        ),
    )

    runner = BenchmarkRunner(config)
    report = runner.run()

    assert report.name == "test-run"
    assert len(report.results) == 2
    clean_res = report.results[0]
    assert clean_res.model == "fake-detector"
    assert clean_res.condition == "clean"
    assert clean_res.metrics["precision"] == pytest.approx(1.0)
    assert clean_res.metrics["recall"] == pytest.approx(1.0)
    assert clean_res.metrics["f1"] == pytest.approx(1.0)

    # 4. Check Markdown formatting
    md = report.to_markdown()
    assert "# PortraitBench Report: test-run" in md
    assert "fake-detector" in md
    assert "clean" in md
    assert "jpeg_heavy" in md


def test_portraitbench_cli_run_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (60, 60), color="blue").save(img_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "cli-bench-ds",
                "images": [{"path": "img.jpg", "faces": [{"box": [10, 10, 50, 50]}]}],
            }
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "bench_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "cli-bench-suite",
                "models": ["fake-detector"],
                "datasets": [
                    {
                        "name": "cli-bench-ds",
                        "track": "detection",
                        "manifest": str(manifest_path),
                        "root": str(tmp_path),
                    }
                ],
                "degradations": [{"name": "clean", "steps": []}],
            }
        ),
        encoding="utf-8",
    )

    fake_detector = StubDetector(boxes=[BoundingBox(10.0, 10.0, 50.0, 50.0)], name="fake-detector")
    import portraitkit.bench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "build_detector", lambda *args, **kwargs: fake_detector)

    out_report = tmp_path / "out_report.json"
    exit_code = bench_main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--output",
            str(out_report),
            "--format",
            "json",
        ]
    )

    assert exit_code == EXIT_OK
    assert out_report.exists()

    # Test report command
    exit_code_report = bench_main(["report", str(out_report), "--format", "markdown"])
    assert exit_code_report == EXIT_OK


def test_portraitbench_cli_degradations_listing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = bench_main(["degradations"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "jpeg_compression" in out
    assert "gaussian_blur" in out
    assert "motion_blur" in out
