"""OFIQ stays an optional, checksum-pinned subprocess boundary."""

from __future__ import annotations

import hashlib
import io
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from portraitkit.config import Settings
from portraitkit.crop import ofiq
from portraitkit.crop.ofiq import (
    OfiqIntegrityError,
    OfiqNotAvailableError,
    OfiqOutputError,
    OfiqPackageSpec,
    OfiqScorer,
    parse_ofiq_csv,
    resolve_reference_ofiq,
)
from portraitkit.errors import OfiqExecutionError

VERSION = "9.9.9"
REVISION = "1" * 40
MODEL_DIRECTORIES = (
    "expression_neutrality",
    "face_detection",
    "face_landmark_estimation",
    "face_occlusion_segmentation",
    "face_parsing",
    "head_pose_estimation",
    "no_compression_artifacts",
    "sharpness",
    "unified_quality_score",
)


def settings(tmp_path: Path, *, allow_download: bool = False) -> Settings:
    return Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        ofiq_dir=tmp_path / "models" / "ofiq",
        allow_download=allow_download,
    )


def package_bytes(*, malicious: bool = False) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("OFIQ-Release/README.md", "fixture")
        archive.writestr("OFIQ-Release/source/LICENSE.md", "fixture license")
        archive.writestr(
            "OFIQ-Release/source/Version.txt",
            "\n".join(
                [
                    "VERSION_MAJOR 9",
                    "VERSION_MINOR 9",
                    "VERSION_PATCH 9",
                    "VERSION_SUFFIX",
                ]
            ),
        )
        archive.writestr("OFIQ-Release/source/data/ofiq_config.jaxn", "{}")
        archive.writestr("OFIQ-Release/releases/win64/OFIQSampleApp.exe", b"fake exe")
        archive.writestr("OFIQ-Release/releases/win64/ofiq_lib.dll", b"fake dll")
        for name in MODEL_DIRECTORIES:
            archive.writestr(f"OFIQ-Release/source/data/models/{name}/fixture.bin", name)
        archive.writestr("OFIQ-Release/source/data/tests/images/public.png", b"png")
        if malicious:
            archive.writestr(
                "OFIQ-Release/source/data/models/../../../../escaped.txt", "must not escape"
            )
    return payload.getvalue()


def make_spec(payload: bytes) -> OfiqPackageSpec:
    return OfiqPackageSpec(
        version=VERSION,
        source_revision=REVISION,
        url="https://example.invalid/ofiq.zip",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def cache_package(tmp_path: Path, payload: bytes, spec: OfiqPackageSpec) -> Settings:
    resolved = settings(tmp_path)
    resolved.ofiq_dir.mkdir(parents=True)
    (resolved.ofiq_dir / f"OFIQ-PrecompiledBinaries-{spec.version}.zip").write_bytes(payload)
    return resolved


def install_fixture(tmp_path: Path):
    payload = package_bytes()
    spec = make_spec(payload)
    return resolve_reference_ofiq(
        cache_package(tmp_path, payload, spec),
        allow_download=False,
        package=spec,
        platform_key="win64",
    )


def test_package_spec_rejects_unpinned_identity() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        OfiqPackageSpec(VERSION, REVISION, "https://example.invalid/x", "short", 1)


def test_offline_resolution_names_the_fetch_command(tmp_path: Path) -> None:
    payload = package_bytes()
    with pytest.raises(OfiqNotAvailableError, match="portraitkit ofiq fetch"):
        resolve_reference_ofiq(
            settings(tmp_path),
            allow_download=False,
            package=make_spec(payload),
            platform_key="win64",
        )


def test_verified_archive_is_selectively_installed(tmp_path: Path) -> None:
    installation = install_fixture(tmp_path)

    assert installation.executable.read_bytes() == b"fake exe"
    assert installation.config.read_text(encoding="utf-8") == "{}"
    assert installation.conformance_images.joinpath("public.png").is_file()
    assert not installation.root.joinpath("source", "unrelated").exists()
    installation.validate()


def test_cached_archive_must_match_its_digest(tmp_path: Path) -> None:
    payload = package_bytes()
    spec = make_spec(payload)
    resolved = cache_package(tmp_path, payload + b"tampered", spec)

    with pytest.raises(OfiqIntegrityError, match="wrong size"):
        resolve_reference_ofiq(resolved, allow_download=False, package=spec, platform_key="win64")


def test_archive_cannot_escape_the_staging_directory(tmp_path: Path) -> None:
    payload = package_bytes(malicious=True)
    spec = make_spec(payload)

    with pytest.raises(OfiqIntegrityError, match="unsafe path"):
        resolve_reference_ofiq(
            cache_package(tmp_path, payload, spec),
            allow_download=False,
            package=spec,
            platform_key="win64",
        )
    assert not tmp_path.joinpath("escaped.txt").exists()


@pytest.mark.parametrize(
    "native_header",
    ["UnifiedQualityScore", "UnifiedQualityScore.native"],
)
def test_parser_accepts_reference_and_later_headers(native_header: str) -> None:
    report = f"Filename;{native_header};UnifiedQualityScore.scalar;\nface.png;12.5;87;\n"

    row = parse_ofiq_csv(report)[0]

    assert row.filename == "face.png"
    assert row.measurements[0].name == "UnifiedQualityScore"
    assert row.measurements[0].native_score == pytest.approx(12.5)
    assert row.measurements[0].scalar_score == pytest.approx(87.0)


def test_parser_preserves_failure_to_assess() -> None:
    row = parse_ofiq_csv("Filename;HeadSize;HeadSize.scalar;\nface.png;0;-1;\n")[0]

    assert not row.measurements[0].assessed


def test_parser_rejects_mismatched_native_and_scalar_columns() -> None:
    with pytest.raises(OfiqOutputError, match="native/scalar"):
        parse_ofiq_csv("Filename;HeadSize;\nface.png;1;\n")


def test_parser_rejects_an_empty_score() -> None:
    with pytest.raises(OfiqOutputError, match="is empty"):
        parse_ofiq_csv("Filename;HeadSize;HeadSize.scalar;\nface.png;;1;\n")


def test_parser_ignores_the_runtime_metadata_column() -> None:
    row = parse_ofiq_csv(
        "Filename;HeadSize;HeadSize.scalar;assessment_time_in_ms;\nface.png;1;90;123;\n"
    )[0]

    assert [measurement.name for measurement in row.measurements] == ["HeadSize"]


def test_scorer_invokes_an_argument_list_and_parses_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = install_fixture(tmp_path)
    image = tmp_path / "input portrait.png"
    image.write_bytes(b"fixture")
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        report = Path(command[command.index("-o") + 1])
        report.write_text(
            f"Filename;UnifiedQualityScore;UnifiedQualityScore.scalar;\n{image};12.5;87;\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(ofiq.subprocess, "run", fake_run)

    result = OfiqScorer(installation).score(image)[0]

    assert observed["command"][0] == str(installation.executable)
    assert observed["command"][1:] == [
        "-c",
        str(installation.config),
        "-i",
        str(image),
        "-o",
        observed["command"][-1],
    ]
    assert "shell" not in observed["kwargs"]
    assert result.image == image
    assert result.measurement("UnifiedQualityScore").scalar_score == 87
    assert result.provenance.version == VERSION
    assert len(result.provenance.models_sha256) == 64


def test_scorer_rejects_a_nonzero_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = install_fixture(tmp_path)
    image = tmp_path / "face.png"
    image.write_bytes(b"fixture")

    def fail(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="bad config")

    monkeypatch.setattr(ofiq.subprocess, "run", fail)

    with pytest.raises(OfiqExecutionError, match=r"code 3.*bad config"):
        OfiqScorer(installation).score(image)


def test_comparison_reports_scalar_deltas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installation = install_fixture(tmp_path)

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        report = Path(command[command.index("-o") + 1])
        input_dir = Path(command[command.index("-i") + 1])
        report.write_text(
            "Filename;HeadSize;HeadSize.scalar;\n"
            f"{input_dir / 'before.png'};1;40;\n"
            f"{input_dir / 'after.png'};2;75;\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ofiq.subprocess, "run", fake_run)
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)

    comparison = OfiqScorer(installation).compare(pixels, pixels)

    assert comparison.scalar_deltas["HeadSize"] == pytest.approx(35.0)
