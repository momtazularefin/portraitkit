"""Command-line interface for the PortraitBench benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from portraitkit import __version__
from portraitkit.bench.config import load_benchmark_config
from portraitkit.bench.report import BenchmarkReport, BenchmarkRunResult
from portraitkit.bench.runner import BenchmarkRunner
from portraitkit.errors import PortraitKitError

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portraitbench",
        description="Standardized benchmark harness and leaderboard runner for PortraitKit.",
    )
    parser.add_argument("--version", action="version", version=f"portraitbench {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_cmd = subcommands.add_parser("run", help="execute a benchmark suite from config")
    run_cmd.add_argument(
        "--config", "-c", type=Path, required=True, help="benchmark configuration JSON"
    )
    run_cmd.add_argument("--output", "-o", type=Path, help="save result report to this path")
    run_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="stdout rendering format (default: markdown)",
    )
    run_cmd.add_argument(
        "--offline",
        action="store_true",
        help="never download models or datasets from the network",
    )

    report_cmd = subcommands.add_parser("report", help="render an existing benchmark report")
    report_cmd.add_argument("report", type=Path, help="path to the JSON report")
    report_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="rendering format",
    )

    subcommands.add_parser("degradations", help="list available degradation transforms")

    return parser


def _command_run(arguments: argparse.Namespace, stream: Any) -> int:
    cfg = load_benchmark_config(arguments.config)
    runner = BenchmarkRunner(cfg, allow_download=not arguments.offline)
    report = runner.run()

    if arguments.output:
        if arguments.output.suffix.lower() == ".json":
            report.write(arguments.output)
        else:
            arguments.output.write_text(report.to_markdown(), encoding="utf-8")

    if arguments.format == "json":
        print(json.dumps(report.to_dict(), indent=2), file=stream)
    else:
        print(report.to_markdown(), file=stream)

    return EXIT_OK


def _command_report(arguments: argparse.Namespace, stream: Any) -> int:
    path = Path(arguments.report)
    if not path.exists():
        msg = f"report file {str(path)!r} does not exist"
        raise PortraitKitError(msg)

    data = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for item in data.get("results", []):
        runs.append(
            BenchmarkRunResult(
                model=item["model"],
                dataset=item["dataset"],
                track=item["track"],
                condition=item["condition"],
                metrics=item.get("metrics", {}),
                latency_mean_ms=item.get("latency_mean_ms", 0.0),
                latency_p95_ms=item.get("latency_p95_ms", 0.0),
                throughput_fps=item.get("throughput_fps", 0.0),
                samples_evaluated=item.get("samples_evaluated", 0),
                status=item.get("status", "ok"),
                error=item.get("error"),
            )
        )

    report = BenchmarkReport(
        name=data.get("benchmark", path.stem),
        results=tuple(runs),
        timestamp=data.get("timestamp", ""),
    )

    if arguments.format == "json":
        print(json.dumps(report.to_dict(), indent=2), file=stream)
    else:
        print(report.to_markdown(), file=stream)

    return EXIT_OK


def _command_degradations(arguments: argparse.Namespace, stream: Any) -> int:
    print("PortraitBench Parameterized Degradation Suite:", file=stream)
    print("  - jpeg_compression: quality factor [1..100]", file=stream)
    print("  - gaussian_blur: sigma [>0], kernel_size [odd integer]", file=stream)
    print("  - motion_blur: kernel_size [>=3], angle_degrees [float]", file=stream)
    print("  - low_light: factor [>0], gamma [>0]", file=stream)
    print("  - gaussian_noise: std [>=0], seed [int]", file=stream)
    print("  - downscale: scale factor [0..1]", file=stream)
    print("  - cluttered_background: pattern ['checkerboard'], frequency [int]", file=stream)
    return EXIT_OK


_COMMANDS = {
    "run": _command_run,
    "report": _command_report,
    "degradations": _command_degradations,
}


def main(argv: Sequence[str] | None = None, stream: Any = None) -> int:
    output = stream if stream is not None else sys.stdout
    arguments = build_parser().parse_args(argv)
    try:
        return _COMMANDS[arguments.command](arguments, output)
    except PortraitKitError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
