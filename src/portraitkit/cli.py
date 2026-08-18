"""Command-line interface.

Four verbs, each mapping to one thing a user actually wants: see which models exist and
what they cost you legally, fetch one, run detection, or score the stage against ground
truth. Everything is available as JSON so the CLI composes with other tools rather than
only being read by a person.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from portraitkit import __version__
from portraitkit.config import load_settings
from portraitkit.detection.base import DetectorConfig
from portraitkit.detection.selection import SelectionStrategy
from portraitkit.detection.stage import DetectionStage, StageConfig, build_detector
from portraitkit.errors import PortraitKitError
from portraitkit.eval.annotations import load_annotations
from portraitkit.eval.runner import evaluate_detection
from portraitkit.imaging.io import load_image
from portraitkit.models.registry import DEFAULT_DETECTOR, MODELS, model_names
from portraitkit.models.store import cached_path, is_cached, resolve_model
from portraitkit.types import DetectionResult

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_ERROR = 1

# DetectorConfig uses slots, so its defaults are only reachable from an instance;
# reading them off the class would yield a descriptor rather than a number.
_DEFAULT_DETECTOR_CONFIG = DetectorConfig()


def _detector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_DETECTOR,
        choices=model_names(),
        help=f"detector to use (default: {DEFAULT_DETECTOR})",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=_DEFAULT_DETECTOR_CONFIG.score_threshold,
        help="minimum detection confidence",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=_DEFAULT_DETECTOR_CONFIG.nms_iou_threshold,
        help="overlap above which a duplicate detection is suppressed",
    )
    parser.add_argument(
        "--selection",
        default=SelectionStrategy.LARGEST.value,
        choices=[strategy.value for strategy in SelectionStrategy],
        help="rule for choosing the primary subject",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download; fail if the model is not already cached",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="portraitkit",
        description="Portrait processing pipeline with stage-by-stage evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"portraitkit {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    models = subcommands.add_parser("models", help="list registered models and their licenses")
    models.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    fetch = subcommands.add_parser("fetch", help="download a model into the local cache")
    fetch.add_argument("model", choices=model_names(), help="model to fetch")

    detect = subcommands.add_parser("detect", help="detect faces in one or more images")
    detect.add_argument("images", nargs="+", type=Path, help="image files to process")
    detect.add_argument("--json", action="store_true", help="emit JSON instead of text")
    detect.add_argument("--output", type=Path, help="write JSON results to this file")
    _detector_options(detect)

    evaluate = subcommands.add_parser("evaluate", help="score the stage against ground truth")
    evaluate.add_argument("manifest", type=Path, help="annotation manifest to evaluate against")
    evaluate.add_argument("--root", type=Path, help="directory image paths are relative to")
    evaluate.add_argument(
        "--iou-threshold", type=float, default=0.5, help="overlap required to count as a match"
    )
    evaluate.add_argument("--output", type=Path, help="write the JSON report to this file")
    _detector_options(evaluate)

    return parser


def _build_stage(arguments: argparse.Namespace) -> DetectionStage:
    detector = build_detector(
        arguments.model,
        config=DetectorConfig(
            score_threshold=arguments.score_threshold,
            nms_iou_threshold=arguments.nms_iou_threshold,
        ),
        allow_download=False if arguments.offline else None,
    )
    return DetectionStage(detector, StageConfig(selection=SelectionStrategy(arguments.selection)))


def _result_to_dict(path: Path, result: DetectionResult) -> dict[str, Any]:
    primary = result.primary
    return {
        "path": str(path),
        "status": str(result.status),
        "detector": result.detector,
        "image_size": list(result.image_size.as_tuple()),
        "face_count": result.face_count,
        "duration_ms": round(result.duration_ms, 3),
        "diagnostics": [str(item) for item in result.diagnostics],
        "primary": None
        if primary is None
        else {
            "box": [round(value, 2) for value in primary.box.as_tuple()],
            "score": round(primary.score, 4),
            "landmarks": None
            if primary.landmarks is None
            else [
                [round(point.x, 2), round(point.y, 2)] for point in primary.landmarks.as_points()
            ],
            "roll_degrees": None
            if primary.landmarks is None
            else round(primary.landmarks.roll_degrees, 2),
        },
    }


def _print_result(path: Path, result: DetectionResult, stream: Any) -> None:
    primary = result.primary
    if primary is None:
        print(f"{path}: {result.status} ({result.duration_ms:.1f} ms)", file=stream)
    else:
        box = primary.box
        print(
            f"{path}: {result.face_count} face(s), primary "
            f"[{box.x1:.0f}, {box.y1:.0f}, {box.x2:.0f}, {box.y2:.0f}] "
            f"score {primary.score:.3f} ({result.duration_ms:.1f} ms)",
            file=stream,
        )
        if primary.landmarks is not None:
            print(f"    roll {primary.landmarks.roll_degrees:+.1f} deg", file=stream)
    if result.diagnostics:
        print(
            f"    diagnostics: {', '.join(str(item) for item in result.diagnostics)}", file=stream
        )


def _command_models(arguments: argparse.Namespace, stream: Any) -> int:
    settings = load_settings()
    entries = []
    for name, spec in MODELS.items():
        entries.append(
            {
                "name": name,
                "default": name == DEFAULT_DETECTOR,
                "cached": is_cached(spec, settings),
                "path": str(cached_path(spec, settings)),
                "size_bytes": spec.size_bytes,
                "license": spec.license,
                "license_url": spec.license_url,
                "permits_commercial_use": spec.permits_commercial_use,
                "upstream": spec.upstream,
                "notes": spec.notes,
            }
        )

    if arguments.json:
        print(json.dumps({"models": entries}, indent=2), file=stream)
        return EXIT_OK

    for entry in entries:
        marker = "*" if entry["default"] else " "
        state = "cached" if entry["cached"] else "not cached"
        commercial = "commercial ok" if entry["permits_commercial_use"] else "RESEARCH ONLY"
        size_mib = entry["size_bytes"] / (1024 * 1024)
        print(
            f"{marker} {entry['name']:<18} {size_mib:>7.2f} MiB  {state:<10} "
            f"{entry['license']} ({commercial})",
            file=stream,
        )
    print("\n* default. Weights carry their own licenses, separate from this project.", file=stream)
    return EXIT_OK


def _command_fetch(arguments: argparse.Namespace, stream: Any) -> int:
    path = resolve_model(arguments.model, allow_download=True)
    print(f"{arguments.model} ready at {path}", file=stream)
    return EXIT_OK


def _command_detect(arguments: argparse.Namespace, stream: Any) -> int:
    stage = _build_stage(arguments)
    payload = []
    for path in arguments.images:
        result = stage.run(load_image(path))
        payload.append(_result_to_dict(path, result))
        if not arguments.json:
            _print_result(path, result, stream)

    if arguments.json:
        print(json.dumps({"results": payload}, indent=2), file=stream)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps({"results": payload}, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if not arguments.json:
            print(f"wrote {arguments.output}", file=stream)
    return EXIT_OK


def _command_evaluate(arguments: argparse.Namespace, stream: Any) -> int:
    annotations = load_annotations(arguments.manifest, root=arguments.root)
    stage = _build_stage(arguments)
    report = evaluate_detection(stage, annotations, iou_threshold=arguments.iou_threshold)
    summary = report.to_dict()["summary"]

    print(f"dataset {report.dataset} | detector {report.detector}", file=stream)
    print(
        f"images {summary['evaluated_images']} evaluated, {summary['errored_images']} errored",
        file=stream,
    )
    print(
        f"precision {summary['precision']:.4f}  recall {summary['recall']:.4f}  "
        f"f1 {summary['f1']:.4f}",
        file=stream,
    )
    for label, key in (("mean IoU", "mean_iou"), ("landmark NME", "mean_landmark_error")):
        value = summary[key]
        print(f"{label}: {'n/a' if value is None else f'{value:.4f}'}", file=stream)
    accuracy = summary["primary_accuracy"]
    print(
        f"primary selection: {'not annotated' if accuracy is None else f'{accuracy:.4f}'}",
        file=stream,
    )

    if arguments.output:
        report.write(arguments.output)
        print(f"wrote {arguments.output}", file=stream)
    return EXIT_OK


_COMMANDS = {
    "models": _command_models,
    "fetch": _command_fetch,
    "detect": _command_detect,
    "evaluate": _command_evaluate,
}


def main(argv: Sequence[str] | None = None, stream: Any = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    output = stream if stream is not None else sys.stdout
    arguments = build_parser().parse_args(argv)
    try:
        return _COMMANDS[arguments.command](arguments, output)
    except PortraitKitError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
