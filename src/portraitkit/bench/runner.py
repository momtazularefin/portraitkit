"""Benchmark execution runner for PortraitBench."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import cv2
import numpy as np

from portraitkit.bench.config import BenchmarkConfig, DatasetSpec, DegradationSpec
from portraitkit.bench.degradations import apply_degradations, build_degradation
from portraitkit.bench.report import BenchmarkReport, BenchmarkRunResult
from portraitkit.detection.stage import DetectionStage, build_detector
from portraitkit.eval.annotations import load_annotations
from portraitkit.eval.matching import match_detections
from portraitkit.eval.matting import _load_mask
from portraitkit.eval.matting_annotations import load_matting_annotations
from portraitkit.eval.metrics import DetectionCounts, normalized_landmark_error
from portraitkit.imaging.io import load_image
from portraitkit.matting.base import build_matter
from portraitkit.matting.metrics import matting_metrics
from portraitkit.matting.stage import MattingStage

__all__ = ["BenchmarkRunner", "run_benchmark"]


class BenchmarkRunner:
    """Orchestrates multi-model, multi-dataset, and multi-degradation benchmarking."""

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        allow_download: bool = False,
    ) -> None:
        self.config = config
        self.allow_download = allow_download

    def run(self) -> BenchmarkReport:
        """Execute the configured benchmark suite.

        Returns:
            A complete :class:`BenchmarkReport`.
        """
        results: list[BenchmarkRunResult] = []

        for dataset_spec in self.config.datasets:
            if dataset_spec.track == "detection":
                runs = self._run_detection_dataset(dataset_spec)
                results.extend(runs)
            elif dataset_spec.track == "matting":
                runs = self._run_matting_dataset(dataset_spec)
                results.extend(runs)

        return BenchmarkReport(name=self.config.name, results=tuple(results))

    def _run_detection_dataset(self, dataset_spec: DatasetSpec) -> list[BenchmarkRunResult]:
        runs: list[BenchmarkRunResult] = []
        try:
            annotations = load_annotations(dataset_spec.manifest_path, root=dataset_spec.root_path)
        except Exception as error:
            for model_name in self.config.models:
                runs.append(
                    BenchmarkRunResult(
                        model=model_name,
                        dataset=dataset_spec.name,
                        track="detection",
                        condition="all",
                        metrics={},
                        latency_mean_ms=0.0,
                        latency_p95_ms=0.0,
                        throughput_fps=0.0,
                        samples_evaluated=0,
                        status="error",
                        error=f"failed to load annotations: {error}",
                    )
                )
            return runs

        for model_name in self.config.models:
            try:
                detector = build_detector(model_name, allow_download=self.allow_download)
                stage = DetectionStage(detector)
            except Exception as error:
                runs.append(
                    BenchmarkRunResult(
                        model=model_name,
                        dataset=dataset_spec.name,
                        track="detection",
                        condition="all",
                        metrics={},
                        latency_mean_ms=0.0,
                        latency_p95_ms=0.0,
                        throughput_fps=0.0,
                        samples_evaluated=0,
                        status="error",
                        error=f"failed to build detector {model_name!r}: {error}",
                    )
                )
                continue

            for deg_spec in self.config.degradations:
                run_res = self._eval_detection_condition(
                    stage, annotations, dataset_spec.name, deg_spec
                )
                runs.append(run_res)

        return runs

    def _eval_detection_condition(
        self,
        stage: DetectionStage,
        annotations: Any,
        dataset_name: str,
        deg_spec: DegradationSpec,
    ) -> BenchmarkRunResult:
        degradations = [build_degradation(step) for step in deg_spec.steps]
        latencies: list[float] = []
        totals = DetectionCounts()
        nme_list: list[float] = []
        samples_evaluated = 0

        for item in annotations.images:
            try:
                image = load_image(item.path)
            except Exception:
                continue

            pixels = (
                apply_degradations(image.pixels, degradations) if degradations else image.pixels
            )

            started = time.perf_counter()
            result = stage.run(pixels)
            duration_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(duration_ms)
            samples_evaluated += 1

            truth_boxes = [f.box for f in item.faces]
            matched = match_detections(truth_boxes, result.faces, iou_threshold=0.5)
            totals = totals + DetectionCounts(
                true_positives=matched.true_positives,
                false_positives=matched.false_positives,
                false_negatives=matched.false_negatives,
            )

            for match in matched.matches:
                exp_lm = item.faces[match.truth_index].landmarks
                act_lm = result.faces[match.prediction_index].landmarks
                if exp_lm and act_lm:
                    with contextlib.suppress(ValueError):
                        nme_list.append(normalized_landmark_error(exp_lm, act_lm))

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

        prec = (
            totals.true_positives / (totals.true_positives + totals.false_positives)
            if (totals.true_positives + totals.false_positives) > 0
            else 0.0
        )
        rec = (
            totals.true_positives / (totals.true_positives + totals.false_negatives)
            if (totals.true_positives + totals.false_negatives) > 0
            else 0.0
        )
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        mean_nme = float(np.mean(nme_list)) if nme_list else 0.0

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "nme": mean_nme,
            "true_positives": float(totals.true_positives),
            "false_positives": float(totals.false_positives),
            "false_negatives": float(totals.false_negatives),
        }

        return BenchmarkRunResult(
            model=stage.detector.name,
            dataset=dataset_name,
            track="detection",
            condition=deg_spec.name,
            metrics=metrics,
            latency_mean_ms=mean_lat,
            latency_p95_ms=p95_lat,
            throughput_fps=fps,
            samples_evaluated=samples_evaluated,
            status="ok",
        )

    def _run_matting_dataset(self, dataset_spec: DatasetSpec) -> list[BenchmarkRunResult]:
        runs: list[BenchmarkRunResult] = []
        try:
            annotations = load_matting_annotations(
                dataset_spec.manifest_path, root=dataset_spec.root_path
            )
        except Exception as error:
            for model_name in self.config.models:
                runs.append(
                    BenchmarkRunResult(
                        model=model_name,
                        dataset=dataset_spec.name,
                        track="matting",
                        condition="all",
                        metrics={},
                        latency_mean_ms=0.0,
                        latency_p95_ms=0.0,
                        throughput_fps=0.0,
                        samples_evaluated=0,
                        status="error",
                        error=f"failed to load matting annotations: {error}",
                    )
                )
            return runs

        for model_name in self.config.models:
            try:
                matter = build_matter(model_name, allow_download=self.allow_download)
                stage = MattingStage(matter)
            except Exception as error:
                runs.append(
                    BenchmarkRunResult(
                        model=model_name,
                        dataset=dataset_spec.name,
                        track="matting",
                        condition="all",
                        metrics={},
                        latency_mean_ms=0.0,
                        latency_p95_ms=0.0,
                        throughput_fps=0.0,
                        samples_evaluated=0,
                        status="error",
                        error=f"failed to build matter {model_name!r}: {error}",
                    )
                )
                continue

            for deg_spec in self.config.degradations:
                run_res = self._eval_matting_condition(
                    stage, annotations, dataset_spec.name, deg_spec
                )
                runs.append(run_res)

        return runs

    def _eval_matting_condition(
        self,
        stage: MattingStage,
        annotations: Any,
        dataset_name: str,
        deg_spec: DegradationSpec,
    ) -> BenchmarkRunResult:
        degradations = [build_degradation(step) for step in deg_spec.steps]
        latencies: list[float] = []
        sads: list[float] = []
        mses: list[float] = []
        grads: list[float] = []
        conns: list[float] = []
        samples_evaluated = 0

        for sample in annotations.samples:
            try:
                image = load_image(sample.image_path)
                truth_alpha = _load_mask(sample.alpha_path)
            except Exception:
                continue

            pixels = (
                apply_degradations(image.pixels, degradations) if degradations else image.pixels
            )

            started = time.perf_counter()
            result = stage.run(pixels)
            duration_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(duration_ms)
            samples_evaluated += 1

            pred_alpha = result.alpha_matte
            if pred_alpha.shape != truth_alpha.shape:
                truth_alpha = cv2.resize(
                    truth_alpha,
                    (pred_alpha.shape[1], pred_alpha.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            unknown_mask = None
            if sample.trimap_path and sample.trimap_path.exists():
                try:
                    trimap = cv2.imread(str(sample.trimap_path), cv2.IMREAD_GRAYSCALE)
                    if trimap is not None:
                        unknown_mask = (trimap > 0) & (trimap < 255)
                except Exception:
                    unknown_mask = None

            m = matting_metrics(pred_alpha, truth_alpha, unknown=unknown_mask)
            sads.append(m.sad)
            mses.append(m.mse)
            grads.append(m.gradient)
            conns.append(m.connectivity)

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

        metrics = {
            "sad": float(np.mean(sads)) if sads else 0.0,
            "mse": float(np.mean(mses)) if mses else 0.0,
            "grad": float(np.mean(grads)) if grads else 0.0,
            "conn": float(np.mean(conns)) if conns else 0.0,
        }

        return BenchmarkRunResult(
            model=stage.matter.name,
            dataset=dataset_name,
            track="matting",
            condition=deg_spec.name,
            metrics=metrics,
            latency_mean_ms=mean_lat,
            latency_p95_ms=p95_lat,
            throughput_fps=fps,
            samples_evaluated=samples_evaluated,
            status="ok",
        )


def run_benchmark(
    config: BenchmarkConfig | str, *, allow_download: bool = False
) -> BenchmarkReport:
    """Convenience helper to load and run a benchmark configuration."""
    from portraitkit.bench.config import load_benchmark_config

    cfg = load_benchmark_config(config) if isinstance(config, str) else config
    runner = BenchmarkRunner(cfg, allow_download=allow_download)
    return runner.run()
