"""PortraitBench: standardized evaluation benchmark and degradation suite."""

from portraitkit.bench.config import (
    BenchmarkConfig,
    DatasetSpec,
    DegradationSpec,
    load_benchmark_config,
)
from portraitkit.bench.degradations import (
    ClutteredBackground,
    Degradation,
    Downscale,
    GaussianBlur,
    GaussianNoise,
    JpegCompression,
    LowLight,
    MotionBlur,
    apply_degradations,
    build_degradation,
)
from portraitkit.bench.report import BenchmarkReport, BenchmarkRunResult
from portraitkit.bench.runner import BenchmarkRunner, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "ClutteredBackground",
    "DatasetSpec",
    "Degradation",
    "DegradationSpec",
    "Downscale",
    "GaussianBlur",
    "GaussianNoise",
    "JpegCompression",
    "LowLight",
    "MotionBlur",
    "apply_degradations",
    "build_degradation",
    "load_benchmark_config",
    "run_benchmark",
]
