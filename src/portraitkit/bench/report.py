"""Results reporting and leaderboard generation for PortraitBench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["BenchmarkReport", "BenchmarkRunResult"]


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Outcome for one (model, dataset, degradation) benchmark cell."""

    model: str
    dataset: str
    track: str
    condition: str
    metrics: dict[str, float]
    latency_mean_ms: float
    latency_p95_ms: float
    throughput_fps: float
    samples_evaluated: int
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "track": self.track,
            "condition": self.condition,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
            "samples_evaluated": self.samples_evaluated,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregated benchmark report over all models and conditions."""

    name: str
    results: tuple[BenchmarkRunResult, ...]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(UTC).isoformat(),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.name,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Render results as a clean GitHub-flavored markdown table."""
        lines = [
            f"# PortraitBench Report: {self.name}",
            "",
            f"Generated at: `{self.timestamp}`",
            "",
        ]

        # Group by track
        tracks: dict[str, list[BenchmarkRunResult]] = {}
        for r in self.results:
            tracks.setdefault(r.track, []).append(r)

        for track, runs in tracks.items():
            lines.append(f"## Track: {track.capitalize()}")
            lines.append("")
            if track == "detection":
                lines.append(
                    "| Model | Dataset | Condition | Precision | Recall | F1 | NME | "
                    "Latency (ms) | FPS |"
                )
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for r in runs:
                    if r.status != "ok":
                        lines.append(
                            f"| {r.model} | {r.dataset} | {r.condition} | "
                            "*error* | *error* | *error* | *error* | - | - |"
                        )
                        continue
                    m = r.metrics
                    p = f"{m.get('precision', 0.0):.3f}"
                    rec = f"{m.get('recall', 0.0):.3f}"
                    f1 = f"{m.get('f1', 0.0):.3f}"
                    nme = f"{m.get('nme', 0.0):.4f}" if "nme" in m else "-"
                    lines.append(
                        f"| {r.model} | {r.dataset} | {r.condition} | "
                        f"{p} | {rec} | {f1} | {nme} | {r.latency_mean_ms:.1f} | "
                        f"{r.throughput_fps:.1f} |"
                    )
            elif track == "matting":
                lines.append(
                    "| Model | Dataset | Condition | SAD | MSE | Grad | Conn | Latency (ms) | FPS |"
                )
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for r in runs:
                    if r.status != "ok":
                        lines.append(
                            f"| {r.model} | {r.dataset} | {r.condition} | "
                            "*error* | *error* | *error* | *error* | - | - |"
                        )
                        continue
                    m = r.metrics
                    sad = f"{m.get('sad', 0.0):.4f}"
                    mse = f"{m.get('mse', 0.0):.6f}"
                    grad = f"{m.get('grad', 0.0):.4f}"
                    conn = f"{m.get('conn', 0.0):.4f}"
                    lines.append(
                        f"| {r.model} | {r.dataset} | {r.condition} | "
                        f"{sad} | {mse} | {grad} | {conn} | "
                        f"{r.latency_mean_ms:.1f} | {r.throughput_fps:.1f} |"
                    )
            else:
                lines.append("| Model | Dataset | Condition | Metrics | Latency (ms) | FPS |")
                lines.append("|---|---|---|---|---|---|")
                for r in runs:
                    metric_str = ", ".join(f"{k}={v:.4f}" for k, v in r.metrics.items())
                    lines.append(
                        f"| {r.model} | {r.dataset} | {r.condition} | "
                        f"{metric_str} | {r.latency_mean_ms:.1f} | {r.throughput_fps:.1f} |"
                    )
            lines.append("")

        return "\n".join(lines)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
