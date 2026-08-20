"""Configuration schemas for PortraitBench runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portraitkit.errors import PortraitKitError

__all__ = [
    "BenchmarkConfig",
    "DatasetSpec",
    "DegradationPreset",
    "DegradationSpec",
    "load_benchmark_config",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Specification of a dataset in the benchmark."""

    name: str
    track: str
    """One of 'detection', 'matting', 'crop'."""

    manifest_path: str
    root_path: str | None = None


@dataclass(frozen=True, slots=True)
class DegradationSpec:
    """Specification of a named degradation condition."""

    name: str
    steps: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Complete configuration for a reproducible benchmark suite."""

    name: str
    models: tuple[str, ...]
    datasets: tuple[DatasetSpec, ...]
    degradations: tuple[DegradationSpec, ...] = field(
        default_factory=lambda: (DegradationSpec(name="clean", steps=()),)
    )
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a benchmark configuration file from JSON.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        A loaded :class:`BenchmarkConfig`.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"benchmark config {str(config_path)!r} does not exist"
        raise PortraitKitError(msg)

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as error:
        msg = f"failed to parse benchmark config {str(config_path)!r}: {error}"
        raise PortraitKitError(msg) from error

    if not isinstance(data, dict):
        msg = "benchmark config root must be a JSON object"
        raise PortraitKitError(msg)

    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        msg = f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION}"
        raise PortraitKitError(msg)

    name = data.get("name", config_path.stem)
    models = tuple(data.get("models", []))

    raw_datasets = data.get("datasets", [])
    datasets: list[DatasetSpec] = []
    for d in raw_datasets:
        datasets.append(
            DatasetSpec(
                name=d["name"],
                track=d.get("track", "detection"),
                manifest_path=d["manifest"],
                root_path=d.get("root"),
            )
        )

    raw_degradations = data.get("degradations", [{"name": "clean", "steps": []}])
    degradations: list[DegradationSpec] = []
    for deg in raw_degradations:
        steps = tuple(deg.get("steps", []))
        degradations.append(DegradationSpec(name=deg["name"], steps=steps))

    return BenchmarkConfig(
        name=name,
        models=models,
        datasets=tuple(datasets),
        degradations=tuple(degradations),
        schema_version=version,
        metadata=data.get("metadata", {}),
    )


# Standard degradation presets
DegradationPreset = {
    "clean": DegradationSpec(name="clean", steps=()),
    "jpeg_heavy": DegradationSpec(name="jpeg_heavy", steps=({"type": "jpeg", "quality": 20},)),
    "motion_blur": DegradationSpec(
        name="motion_blur", steps=({"type": "motion_blur", "kernel_size": 15},)
    ),
    "low_light": DegradationSpec(
        name="low_light", steps=({"type": "low_light", "factor": 0.35, "gamma": 1.4},)
    ),
    "sensor_noise": DegradationSpec(
        name="sensor_noise", steps=({"type": "gaussian_noise", "std": 30.0},)
    ),
    "downscale": DegradationSpec(name="downscale", steps=({"type": "downscale", "scale": 0.25},)),
}
