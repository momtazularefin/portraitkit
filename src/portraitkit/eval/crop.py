"""Aggregate OFIQ evidence for PortraitKit's crop stage.

The evaluator deliberately publishes no per-image rows. It stages temporary before and
after images for OFIQ, then keeps only aggregate quality changes and cryptographic input
provenance. This satisfies the project's identity-sealed reporting boundary while still
making a run reproducible.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from portraitkit.crop.ofiq import CROP_QUALITY_MEASURES, OfiqProvenance, OfiqScorer
from portraitkit.crop.stage import CropStage
from portraitkit.detection.stage import DetectionStage
from portraitkit.errors import ImageLoadError, OfiqOutputError
from portraitkit.imaging.io import load_image
from portraitkit.models.registry import get_model
from portraitkit.models.store import file_digest

__all__ = ["CropQualityAggregate", "CropQualityReport", "evaluate_crop_quality"]


def _rounded(value: float) -> float:
    return round(value, 4)


@dataclass(frozen=True, slots=True)
class CropQualityAggregate:
    """Aggregate before/after scalar scores for one OFIQ quality measure."""

    pairs: int
    before_mean: float
    after_mean: float
    before_median: float
    after_median: float
    mean_delta: float
    median_delta: float
    improved: int
    unchanged: int
    worsened: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "pairs": self.pairs,
            "before_mean": _rounded(self.before_mean),
            "after_mean": _rounded(self.after_mean),
            "before_median": _rounded(self.before_median),
            "after_median": _rounded(self.after_median),
            "mean_delta": _rounded(self.mean_delta),
            "median_delta": _rounded(self.median_delta),
            "improved": self.improved,
            "unchanged": self.unchanged,
            "worsened": self.worsened,
        }


@dataclass(frozen=True, slots=True)
class CropQualityReport:
    """Identity-free aggregate evidence from one public-sample crop run."""

    dataset: str
    dataset_provenance: dict[str, str]
    input_set_sha256: str
    input_count: int
    scored_pairs: int
    conforming_crops: int
    padded_crops: int
    failures: dict[str, int]
    detector: str
    detector_artifact_sha256: str
    crop_preset: str
    measures: dict[str, CropQualityAggregate]
    ofiq: OfiqProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": {"name": self.dataset, **self.dataset_provenance},
            "input_set_sha256": self.input_set_sha256,
            "input_count": self.input_count,
            "scored_pairs": self.scored_pairs,
            "conforming_crops": self.conforming_crops,
            "padded_crops": self.padded_crops,
            "failures": dict(sorted(self.failures.items())),
            "detector": {
                "name": self.detector,
                "artifact_sha256": self.detector_artifact_sha256,
            },
            "crop_preset": self.crop_preset,
            "measures": {name: value.to_dict() for name, value in self.measures.items()},
            "ofiq": self.ofiq.to_dict(),
        }

    def write(self, path: Path) -> None:
        """Write deterministic aggregate JSON without exposing source paths."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _input_set_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        digest.update(f"{index}:".encode())
        content_digest = file_digest(path) if path.is_file() else "missing"
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _aggregate(pairs: list[tuple[float, float]]) -> CropQualityAggregate:
    before = [item[0] for item in pairs]
    after = [item[1] for item in pairs]
    deltas = [after_score - before_score for before_score, after_score in pairs]
    return CropQualityAggregate(
        pairs=len(pairs),
        before_mean=statistics.fmean(before),
        after_mean=statistics.fmean(after),
        before_median=statistics.median(before),
        after_median=statistics.median(after),
        mean_delta=statistics.fmean(deltas),
        median_delta=statistics.median(deltas),
        improved=sum(delta > 0 for delta in deltas),
        unchanged=sum(delta == 0 for delta in deltas),
        worsened=sum(delta < 0 for delta in deltas),
    )


def evaluate_crop_quality(
    detection_stage: DetectionStage,
    crop_stage: CropStage,
    scorer: OfiqScorer,
    images: tuple[Path, ...],
    *,
    dataset: str,
    dataset_provenance: dict[str, str] | None = None,
) -> CropQualityReport:
    """Run detection, crop, and external scoring; return aggregate-only evidence."""
    sources = tuple(Path(path) for path in images)
    if not sources:
        raise ValueError("at least one public sample image is required")

    failures: dict[str, int] = {}
    conforming = 0
    padded = 0
    pairs: list[tuple[str, str]] = []

    def fail(reason: str) -> None:
        failures[reason] = failures.get(reason, 0) + 1

    with tempfile.TemporaryDirectory(prefix="portraitkit-crop-eval-") as raw:
        staging = Path(raw)
        for index, source in enumerate(sources):
            try:
                loaded = load_image(source)
            except ImageLoadError:
                fail("load_error")
                continue
            detection = detection_stage.run(loaded)
            if detection.primary is None:
                fail("no_face")
                continue
            cropped = crop_stage.run(loaded, detection)
            if not cropped.ok or cropped.image is None:
                fail(str(cropped.status))
                continue

            conforming += int(cropped.conforms)
            padded += int(cropped.padded)
            before_name = f"sample-{index:04d}-before.png"
            after_name = f"sample-{index:04d}-after.png"
            Image.fromarray(loaded.pixels).save(staging / before_name)
            Image.fromarray(cropped.image).save(staging / after_name)
            pairs.append((before_name, after_name))

        results = scorer.score(staging) if pairs else ()

    by_name = {result.image.name: result for result in results}
    measure_pairs: dict[str, list[tuple[float, float]]] = {
        name: [] for name in CROP_QUALITY_MEASURES
    }
    for before_name, after_name in pairs:
        try:
            before = {item.name: item for item in by_name[before_name].measurements}
            after = {item.name: item for item in by_name[after_name].measurements}
        except KeyError as error:
            raise OfiqOutputError(
                f"OFIQ omitted a staged crop-evaluation image: {error}"
            ) from error
        for name in CROP_QUALITY_MEASURES:
            if name not in before or name not in after:
                continue
            if before[name].assessed and after[name].assessed:
                measure_pairs[name].append((before[name].scalar_score, after[name].scalar_score))

    detector_name = detection_stage.detector.name
    detector_spec = get_model(detector_name)
    return CropQualityReport(
        dataset=dataset,
        dataset_provenance=dataset_provenance or {},
        input_set_sha256=_input_set_digest(sources),
        input_count=len(sources),
        scored_pairs=len(pairs),
        conforming_crops=conforming,
        padded_crops=padded,
        failures=failures,
        detector=detector_name,
        detector_artifact_sha256=detector_spec.sha256,
        crop_preset=crop_stage.preset.name,
        measures={name: _aggregate(values) for name, values in measure_pairs.items() if values},
        ofiq=scorer.provenance,
    )
