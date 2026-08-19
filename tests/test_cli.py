"""Command-line interface.

Commands that need model weights self-skip when the artifact is not cached, so this file
still exercises argument handling, the model listing, and every error path in CI.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import portraitkit.cli as cli
from portraitkit import __version__
from portraitkit.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from portraitkit.config import load_settings
from portraitkit.models.registry import DEFAULT_DETECTOR, model_names
from portraitkit.models.store import is_cached
from tests.conftest import solid_image


def run(*argv: str) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(list(argv), stream=stream)
    return code, stream.getvalue()


@pytest.fixture
def weights_required() -> None:
    if not is_cached(DEFAULT_DETECTOR, load_settings()):
        pytest.skip(f"{DEFAULT_DETECTOR} is not cached; run 'portraitkit fetch' first")


@pytest.fixture
def blank_image(tmp_path: Path) -> Path:
    path = tmp_path / "blank.png"
    Image.fromarray(solid_image(320, 320), mode="RGB").save(path)
    return path


# --- parser ---------------------------------------------------------------------------


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["polish"])


def test_detector_defaults_are_real_numbers() -> None:
    """Regression: DetectorConfig uses slots, so class-level access yields a descriptor.

    Reading defaults off the class produced a member_descriptor that only failed once a
    detector was actually constructed, well past argument parsing.
    """
    arguments = build_parser().parse_args(["detect", "x.jpg"])

    assert isinstance(arguments.score_threshold, float)
    assert isinstance(arguments.nms_iou_threshold, float)
    assert 0.0 <= arguments.score_threshold <= 1.0


def test_detect_accepts_several_images() -> None:
    arguments = build_parser().parse_args(["detect", "a.jpg", "b.jpg"])

    assert [path.name for path in arguments.images] == ["a.jpg", "b.jpg"]


def test_unknown_model_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main(["detect", "a.jpg", "--model", "nonexistent"])


def test_unknown_selection_strategy_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["detect", "a.jpg", "--selection", "prettiest"])


def test_ofiq_requires_a_nested_command() -> None:
    with pytest.raises(SystemExit):
        main(["ofiq"])


def test_ofiq_crop_evaluation_accepts_public_samples() -> None:
    arguments = build_parser().parse_args(["ofiq", "evaluate-crop", "a.jpg", "b.jpg", "--offline"])

    assert arguments.ofiq_command == "evaluate-crop"
    assert [path.name for path in arguments.images] == ["a.jpg", "b.jpg"]
    assert arguments.preset == "icao-portrait-35x45"
    assert arguments.offline is True


# --- models ---------------------------------------------------------------------------


def test_models_lists_every_entry() -> None:
    code, output = run("models")

    assert code == EXIT_OK
    assert "yunet-2023mar" in output
    assert "scrfd-10g-bnkps" in output


def test_models_marks_the_default() -> None:
    _, output = run("models")

    default_line = next(line for line in output.splitlines() if DEFAULT_DETECTOR in line)
    assert default_line.startswith("*")


def test_models_surfaces_the_research_only_restriction() -> None:
    """An integrator must not have to read upstream docs to learn this."""
    _, output = run("models")

    assert "RESEARCH ONLY" in output
    assert "commercial ok" in output


def test_models_json_is_machine_readable() -> None:
    code, output = run("models", "--json")

    payload = json.loads(output)
    assert code == EXIT_OK
    names = {entry["name"] for entry in payload["models"]}
    assert names == set(model_names())
    for entry in payload["models"]:
        assert "license" in entry
        assert isinstance(entry["permits_commercial_use"], bool)


# --- detect ---------------------------------------------------------------------------


def test_missing_image_reports_an_error_not_a_traceback(
    weights_required: None, tmp_path: Path
) -> None:
    code, _ = run("detect", str(tmp_path / "absent.jpg"))

    assert code == EXIT_ERROR


def test_offline_flag_fails_when_the_model_is_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PORTRAITKIT_MODEL_DIR", str(tmp_path / "empty"))

    code, _ = run("detect", "a.jpg", "--offline")

    assert code == EXIT_ERROR


def test_detect_reports_no_face_on_a_blank_frame(weights_required: None, blank_image: Path) -> None:
    code, output = run("detect", str(blank_image))

    assert code == EXIT_OK
    assert "no_face" in output


def test_detect_json_shape(weights_required: None, blank_image: Path) -> None:
    code, output = run("detect", str(blank_image), "--json")

    payload = json.loads(output)
    assert code == EXIT_OK
    result = payload["results"][0]
    assert result["status"] == "no_face"
    assert result["primary"] is None
    assert result["image_size"] == [320, 320]
    assert result["detector"] == DEFAULT_DETECTOR


def test_detect_writes_an_output_file(
    weights_required: None, blank_image: Path, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "results.json"

    code, _ = run("detect", str(blank_image), "--output", str(target))

    assert code == EXIT_OK
    assert json.loads(target.read_text(encoding="utf-8"))["results"][0]["face_count"] == 0


# --- evaluate -------------------------------------------------------------------------


def test_evaluate_reports_a_bad_manifest(tmp_path: Path) -> None:
    code, _ = run("evaluate", str(tmp_path / "absent.json"))

    assert code == EXIT_ERROR


def test_evaluate_runs_and_writes_a_report(
    weights_required: None, tmp_path: Path, blank_image: Path
) -> None:
    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "blank-set",
                "images": [{"path": blank_image.name, "faces": []}],
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "report.json"

    code, output = run("evaluate", str(manifest), "--output", str(target))

    assert code == EXIT_OK
    assert "blank-set" in output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["dataset"] == "blank-set"
    assert payload["summary"]["evaluated_images"] == 1
    assert payload["summary"]["primary_accuracy"] is None


# --- OFIQ -----------------------------------------------------------------------------


def test_ofiq_fetch_reports_pinned_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = SimpleNamespace(
        version="1.0.3",
        package_sha256="a" * 64,
        to_dict=lambda: {"version": "1.0.3", "package_sha256": "a" * 64},
    )
    installation = SimpleNamespace(
        root=Path("models/ofiq/1.0.3/OFIQ-Release"),
        provenance=lambda: provenance,
    )
    observed: dict[str, object] = {}

    def resolve(**kwargs):
        observed.update(kwargs)
        return installation

    monkeypatch.setattr(cli, "resolve_reference_ofiq", resolve)

    code, output = run("ofiq", "fetch", "--json")

    payload = json.loads(output)
    assert code == EXIT_OK
    assert observed == {"allow_download": True}
    assert payload["provenance"]["version"] == "1.0.3"
    assert payload["installation"].endswith("OFIQ-Release")


def test_ofiq_score_is_offline_capable_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "face.png"
    target = tmp_path / "reports" / "ofiq.json"
    measurement = SimpleNamespace(name="UnifiedQualityScore", native_score=71.0, scalar_score=82.0)
    result = SimpleNamespace(
        image=input_path,
        measurements=(measurement,),
        to_dict=lambda: {
            "image": str(input_path),
            "measurements": {"UnifiedQualityScore": {"scalar_score": 82.0}},
        },
    )
    installation = object()
    observed: dict[str, object] = {}

    def resolve(**kwargs):
        observed.update(kwargs)
        return installation

    class FakeScorer:
        def __init__(self, selected_installation: object):
            assert selected_installation is installation

        def score(self, selected_input: Path):
            assert selected_input == input_path
            return (result,)

    monkeypatch.setattr(cli, "resolve_reference_ofiq", resolve)
    monkeypatch.setattr(cli, "OfiqScorer", FakeScorer)

    code, output = run("ofiq", "score", str(input_path), "--offline", "--output", str(target))

    assert code == EXIT_OK
    assert observed == {"allow_download": False}
    assert "UnifiedQualityScore: scalar 82" in output
    assert json.loads(target.read_text(encoding="utf-8"))["results"][0]["image"] == str(input_path)
