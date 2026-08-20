"""Command-line interface.

Commands cover model discovery, detection evaluation, and the optional external OFIQ
quality referee. Machine-readable output keeps each workflow composable.
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
from portraitkit.crop.ofiq import CROP_QUALITY_MEASURES, OfiqScorer, resolve_reference_ofiq
from portraitkit.crop.presets import preset_names
from portraitkit.crop.stage import CropConfig, CropStage
from portraitkit.detection.base import DetectorConfig
from portraitkit.detection.selection import SelectionStrategy
from portraitkit.detection.stage import DetectionStage, StageConfig, build_detector
from portraitkit.errors import AnnotationError, PortraitKitError
from portraitkit.eval.annotations import load_annotations
from portraitkit.eval.crop import evaluate_crop_quality
from portraitkit.eval.matting import evaluate_matting
from portraitkit.eval.matting_annotations import load_matting_annotations
from portraitkit.eval.runner import evaluate_detection
from portraitkit.eval.samples import resolve_public_samples
from portraitkit.imaging.io import load_image
from portraitkit.matting import (
    MattingStage,
    MattingStageConfig,
    build_matter,
    parse_color,
)
from portraitkit.models.registry import (
    DEFAULT_DETECTOR,
    DEFAULT_MATTER,
    MODELS,
    model_names,
)
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

    ofiq = subcommands.add_parser(
        "ofiq", help="manage and run the pinned external OFIQ quality referee"
    )
    ofiq_commands = ofiq.add_subparsers(dest="ofiq_command", required=True)

    ofiq_fetch = ofiq_commands.add_parser(
        "fetch", help="install the official pinned OFIQ reference package"
    )
    ofiq_fetch.add_argument("--json", action="store_true", help="emit provenance as JSON")

    ofiq_score = ofiq_commands.add_parser(
        "score", help="score one image or an image directory with OFIQ"
    )
    ofiq_score.add_argument("input", type=Path, help="image file or directory to score")
    ofiq_score.add_argument("--offline", action="store_true", help="never download OFIQ")
    ofiq_score.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ofiq_score.add_argument("--output", type=Path, help="write the JSON results to this file")

    ofiq_evaluate = ofiq_commands.add_parser(
        "evaluate-crop", help="measure aggregate OFIQ changes across public sample crops"
    )
    ofiq_evaluate.add_argument("images", nargs="*", type=Path, help="public sample images")
    ofiq_evaluate.add_argument(
        "--manifest", type=Path, help="download and verify a pinned public sample manifest"
    )
    ofiq_evaluate.add_argument(
        "--dataset", default="public-samples", help="identity-free dataset label for the report"
    )
    ofiq_evaluate.add_argument("--preset", choices=preset_names(), default="icao-portrait-35x45")
    ofiq_evaluate.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ofiq_evaluate.add_argument("--output", type=Path, help="write aggregate JSON to this file")
    _detector_options(ofiq_evaluate)

    matte = subcommands.add_parser("matte", help="remove or replace background in images")
    matte.add_argument("images", nargs="+", type=Path, help="image files to process")
    matte.add_argument(
        "--model",
        default=DEFAULT_MATTER,
        choices=model_names(),
        help=f"matting model to use (default: {DEFAULT_MATTER})",
    )
    matte.add_argument(
        "--color",
        default="white",
        help="solid background color (name, hex code like '#ffffff', or 'transparent')",
    )
    matte.add_argument(
        "--transparent",
        action="store_true",
        help="output transparent PNG (equivalent to --color transparent)",
    )
    matte.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="optional binarization threshold in [0, 1]",
    )
    matte.add_argument("--json", action="store_true", help="emit JSON instead of text")
    matte.add_argument("--output", type=Path, help="output directory or file to save result")
    matte.add_argument(
        "--offline",
        action="store_true",
        help="never download; fail if the model is not already cached",
    )

    eval_matte = subcommands.add_parser(
        "evaluate-matting", help="score a matting model against ground truth"
    )
    eval_matte.add_argument(
        "manifest", type=Path, help="matting annotation manifest to evaluate against"
    )
    eval_matte.add_argument(
        "--model",
        default=DEFAULT_MATTER,
        choices=model_names(),
        help=f"matting model to evaluate (default: {DEFAULT_MATTER})",
    )
    eval_matte.add_argument("--root", type=Path, help="directory image paths are relative to")
    eval_matte.add_argument("--json", action="store_true", help="emit JSON instead of text")
    eval_matte.add_argument("--output", type=Path, help="write the JSON report to this file")
    eval_matte.add_argument(
        "--offline",
        action="store_true",
        help="never download; fail if the model is not already cached",
    )

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


def _write_json_output(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _command_ofiq(arguments: argparse.Namespace, stream: Any) -> int:
    if arguments.ofiq_command == "fetch":
        installation = resolve_reference_ofiq(allow_download=True)
        provenance = installation.provenance()
        payload = {
            "installation": str(installation.root),
            "provenance": provenance.to_dict(),
        }
        if arguments.json:
            print(json.dumps(payload, indent=2), file=stream)
        else:
            print(
                f"OFIQ {provenance.version} reference package ready at {installation.root}",
                file=stream,
            )
            print(f"package sha256 {provenance.package_sha256}", file=stream)
        return EXIT_OK

    if arguments.ofiq_command == "score":
        installation = resolve_reference_ofiq(allow_download=False if arguments.offline else None)
        results = OfiqScorer(installation).score(arguments.input)
        payload = {"results": [result.to_dict() for result in results]}
        if arguments.json:
            print(json.dumps(payload, indent=2, allow_nan=False), file=stream)
        else:
            for result in results:
                print(result.image, file=stream)
                measurements = {item.name: item for item in result.measurements}
                for name in CROP_QUALITY_MEASURES:
                    measurement = measurements.get(name)
                    if measurement is not None:
                        print(
                            f"    {name}: scalar {measurement.scalar_score:g} "
                            f"(native {measurement.native_score:g})",
                            file=stream,
                        )
        if arguments.output:
            _write_json_output(payload, arguments.output)
            if not arguments.json:
                print(f"wrote {arguments.output}", file=stream)
        return EXIT_OK

    installation = resolve_reference_ofiq(allow_download=False if arguments.offline else None)
    detection = _build_stage(arguments)
    crop = CropStage(CropConfig(preset=arguments.preset))
    if bool(arguments.images) == bool(arguments.manifest):
        raise AnnotationError("provide either public sample images or --manifest, not both")
    dataset_provenance: dict[str, str] = {}
    if arguments.manifest:
        samples = resolve_public_samples(
            arguments.manifest,
            allow_download=False if arguments.offline else None,
        )
        images = samples.paths
        dataset = samples.manifest.name
        dataset_provenance = {
            "manifest_sha256": samples.manifest_sha256,
            "source_revision": samples.manifest.source_revision,
            "license": samples.manifest.license,
            "license_url": samples.manifest.license_url,
        }
    else:
        images = tuple(arguments.images)
        dataset = arguments.dataset

    report = evaluate_crop_quality(
        detection,
        crop,
        OfiqScorer(installation),
        images,
        dataset=dataset,
        dataset_provenance=dataset_provenance,
    )
    payload = report.to_dict()
    if arguments.json:
        print(json.dumps(payload, indent=2, allow_nan=False), file=stream)
    else:
        print(
            f"{report.dataset}: {report.scored_pairs}/{report.input_count} crops scored; "
            f"{report.conforming_crops} geometry-conforming",
            file=stream,
        )
        for name, aggregate in report.measures.items():
            print(
                f"    {name}: median {aggregate.before_median:g} -> "
                f"{aggregate.after_median:g} ({aggregate.median_delta:+g})",
                file=stream,
            )
    if arguments.output:
        report.write(arguments.output)
        if not arguments.json:
            print(f"wrote {arguments.output}", file=stream)
    return EXIT_OK


def _command_matte(arguments: argparse.Namespace, stream: Any) -> int:
    from PIL import Image

    bg_color = None if arguments.transparent else parse_color(arguments.color)
    matter = build_matter(
        arguments.model,
        allow_download=False if arguments.offline else None,
    )
    stage = MattingStage(
        matter,
        MattingStageConfig(background_color=bg_color, threshold=arguments.threshold),
    )

    payload = []
    for path in arguments.images:
        image = load_image(path)
        result = stage.run(image)

        saved_path = None
        if arguments.output:
            out_target = arguments.output
            if len(arguments.images) == 1 and out_target.suffix.lower() in (
                ".png",
                ".jpg",
                ".jpeg",
            ):
                save_file = out_target
            else:
                out_target.mkdir(parents=True, exist_ok=True)
                ext = ".png" if bg_color is None else ".jpg"
                save_file = out_target / f"{path.stem}_matte{ext}"

            save_file.parent.mkdir(parents=True, exist_ok=True)
            if bg_color is None:
                Image.fromarray(result.image_rgba, mode="RGBA").save(save_file)
            else:
                Image.fromarray(result.image_rgb, mode="RGB").save(save_file)
            saved_path = str(save_file)

        item = {
            "path": str(path),
            "matter": result.matter,
            "image_size": list(result.image_size.as_tuple()),
            "duration_ms": round(result.duration_ms, 3),
            "saved_to": saved_path,
        }
        payload.append(item)

        if not arguments.json:
            save_info = f" -> {saved_path}" if saved_path else ""
            print(
                f"{path}: matting via {result.matter} ({result.duration_ms:.1f} ms){save_info}",
                file=stream,
            )

    if arguments.json:
        print(json.dumps({"results": payload}, indent=2), file=stream)
    return EXIT_OK


def _command_evaluate_matting(arguments: argparse.Namespace, stream: Any) -> int:
    matter = build_matter(
        arguments.model,
        allow_download=False if arguments.offline else None,
    )
    annotations = load_matting_annotations(arguments.manifest, root=arguments.root)
    report = evaluate_matting(matter, annotations)

    if arguments.json:
        print(json.dumps(report.to_dict(), indent=2), file=stream)
    else:
        summary = report.summary
        print(f"dataset {report.dataset} | matter {report.matter}", file=stream)
        print(
            f"samples: {summary.evaluated_samples} evaluated, {summary.errored_samples} errored",
            file=stream,
        )
        print(
            f"mean SAD {summary.mean_sad:.4f}  MSE {summary.mean_mse:.6f}  "
            f"Grad {summary.mean_gradient:.4f}  Conn {summary.mean_connectivity:.4f}",
            file=stream,
        )
        print(
            f"median SAD {summary.median_sad:.4f}  MSE {summary.median_mse:.6f}  "
            f"Grad {summary.median_gradient:.4f}  Conn {summary.median_connectivity:.4f}",
            file=stream,
        )

    if arguments.output:
        report.write(arguments.output)
        if not arguments.json:
            print(f"wrote {arguments.output}", file=stream)
    return EXIT_OK


_COMMANDS = {
    "models": _command_models,
    "fetch": _command_fetch,
    "detect": _command_detect,
    "evaluate": _command_evaluate,
    "ofiq": _command_ofiq,
    "matte": _command_matte,
    "evaluate-matting": _command_evaluate_matting,
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
