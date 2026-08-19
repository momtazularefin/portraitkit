"""Pinned subprocess integration for the external OFIQ quality referee.

PortraitKit intentionally does not link OFIQ into its process.  The official OFIQ
sample application is a stable, language-neutral boundary, keeps the C++ runtime
optional, and lets normal tests remain offline.  The selected package is OFIQ 1.0.3,
which upstream identifies as the latest ISO/IEC 29794-5 reference implementation.

OFIQ evaluates face-image *quality*.  It does not certify PortraitKit's separate
ISO/IEC 39794-5 crop-geometry checks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from portraitkit.config import Settings, load_settings
from portraitkit.errors import (
    OfiqExecutionError,
    OfiqIntegrityError,
    OfiqNotAvailableError,
    OfiqOutputError,
)
from portraitkit.models.store import file_digest

__all__ = [
    "CROP_QUALITY_MEASURES",
    "OFIQ_REFERENCE",
    "OfiqComparison",
    "OfiqInstallation",
    "OfiqMeasurement",
    "OfiqPackageSpec",
    "OfiqProvenance",
    "OfiqResult",
    "OfiqScorer",
    "parse_ofiq_csv",
    "resolve_reference_ofiq",
]

_DOWNLOAD_TIMEOUT_SECONDS: Final = 900
_COPY_CHUNK_BYTES: Final = 1 << 20
_MARKER_NAME: Final = ".portraitkit-ofiq.json"
_METADATA_COLUMNS: Final = {"assessment_time_in_ms"}

CROP_QUALITY_MEASURES: Final = (
    "UnifiedQualityScore",
    "InterEyeDistance",
    "HeadSize",
    "LeftwardCropOfTheFaceImage",
    "RightwardCropOfTheFaceImage",
    "MarginAboveOfTheFaceImage",
    "MarginBelowOfTheFaceImage",
    "HeadPoseRoll",
)

_MODEL_DIRECTORIES: Final = (
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


@dataclass(frozen=True, slots=True)
class OfiqPackageSpec:
    """Immutable identity of an official OFIQ package."""

    version: str
    source_revision: str
    url: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if urlparse(self.url).scheme != "https":
            raise ValueError("OFIQ package URL must use https")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("OFIQ package SHA-256 must be 64 lowercase hexadecimal characters")
        if len(self.source_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.source_revision
        ):
            raise ValueError("OFIQ source revision must be a full Git commit SHA")
        if self.size_bytes <= 0:
            raise ValueError("OFIQ package size must be positive")


OFIQ_REFERENCE = OfiqPackageSpec(
    version="1.0.3",
    source_revision="df8fbb5e4bd8de09ae998ff69bc252f6be4367f8",
    url="https://resources.eulisa.europa.eu/research/OFIQ-PrecompiledBinaries.zip",
    sha256="c8996f37246731d7bd195697da04542b97cdf4e34a5d43ea1ab6a3138c4620ce",
    size_bytes=1_045_715_910,
)


def _platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "win64"
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "ubuntu_x86_64"
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return "ubuntu_arm64"
    if system == "darwin" and machine in {"aarch64", "arm64"}:
        return "macos_arm64"
    msg = f"the official OFIQ {OFIQ_REFERENCE.version} package has no binary for {system}/{machine}"
    raise OfiqNotAvailableError(msg)


def _executable_name(platform_key: str) -> str:
    return "OFIQSampleApp.exe" if platform_key.startswith("win") else "OFIQSampleApp"


def _tree_digest(root: Path) -> str:
    """Hash a directory as relative path plus per-file SHA-256 entries."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_version(path: Path) -> str:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"VERSION_MAJOR", "VERSION_MINOR", "VERSION_PATCH"}:
            values[parts[0]] = parts[1]
    try:
        return ".".join(values[key] for key in ("VERSION_MAJOR", "VERSION_MINOR", "VERSION_PATCH"))
    except KeyError as error:
        raise OfiqIntegrityError(f"could not read an OFIQ version from {path}") from error


@dataclass(frozen=True, slots=True)
class OfiqProvenance:
    """Exact external inputs behind one OFIQ result."""

    version: str
    source_revision: str
    package_sha256: str
    executable_sha256: str
    config_sha256: str
    models_sha256: str
    platform: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "source_revision": self.source_revision,
            "package_sha256": self.package_sha256,
            "executable_sha256": self.executable_sha256,
            "config_sha256": self.config_sha256,
            "models_sha256": self.models_sha256,
            "platform": self.platform,
        }


@dataclass(frozen=True, slots=True)
class OfiqInstallation:
    """Verified paths inside an extracted official package."""

    root: Path
    package: OfiqPackageSpec = OFIQ_REFERENCE
    platform_key: str = ""

    def __post_init__(self) -> None:
        if not self.platform_key:
            object.__setattr__(self, "platform_key", _platform_key())

    @property
    def executable(self) -> Path:
        return self.root / "releases" / self.platform_key / _executable_name(self.platform_key)

    @property
    def config(self) -> Path:
        return self.root / "source" / "data" / "ofiq_config.jaxn"

    @property
    def models(self) -> Path:
        return self.root / "source" / "data" / "models"

    @property
    def version_file(self) -> Path:
        return self.root / "source" / "Version.txt"

    @property
    def conformance_images(self) -> Path:
        return self.root / "source" / "data" / "tests" / "images"

    def validate(self, *, require_marker: bool = True) -> None:
        """Reject incomplete, wrong-version, or unverified installations."""
        for label, path in (
            ("executable", self.executable),
            ("configuration", self.config),
            ("version file", self.version_file),
        ):
            if not path.is_file():
                raise OfiqNotAvailableError(f"OFIQ {label} is missing at {path}")

        actual_version = _read_version(self.version_file)
        if actual_version != self.package.version:
            msg = f"expected OFIQ {self.package.version}, found {actual_version} at {self.root}"
            raise OfiqIntegrityError(msg)

        missing_models = [
            name
            for name in _MODEL_DIRECTORIES
            if not any(path.is_file() for path in (self.models / name).rglob("*"))
        ]
        if missing_models:
            joined = ", ".join(missing_models)
            raise OfiqNotAvailableError(f"OFIQ model directories are missing or empty: {joined}")

        if require_marker:
            marker = self.root / _MARKER_NAME
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                message = f"OFIQ installation marker is missing or invalid at {marker}"
                raise OfiqIntegrityError(message) from error
            if payload.get("package_sha256") != self.package.sha256:
                message = "OFIQ installation marker does not match the pinned package"
                raise OfiqIntegrityError(message)

    def provenance(self) -> OfiqProvenance:
        """Compute tamper-evident provenance for a result."""
        self.validate()
        return OfiqProvenance(
            version=self.package.version,
            source_revision=self.package.source_revision,
            package_sha256=self.package.sha256,
            executable_sha256=file_digest(self.executable),
            config_sha256=file_digest(self.config),
            models_sha256=_tree_digest(self.models),
            platform=self.platform_key,
        )


def _archive_path(settings: Settings, package: OfiqPackageSpec) -> Path:
    return settings.ofiq_dir / f"OFIQ-PrecompiledBinaries-{package.version}.zip"


def _install_root(settings: Settings, package: OfiqPackageSpec) -> Path:
    return settings.ofiq_dir / package.version / "OFIQ-Release"


def _download_package(package: OfiqPackageSpec, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(
        package.url,
        headers={"User-Agent": "portraitkit/0.1 (+https://github.com/momtazularefin/portraitkit)"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as handle,
        ):
            while chunk := response.read(_COPY_CHUNK_BYTES):
                handle.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        message = f"could not download OFIQ {package.version}: {error}"
        raise OfiqNotAvailableError(message) from error

    if partial.stat().st_size != package.size_bytes:
        actual_size = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise OfiqIntegrityError(
            f"downloaded OFIQ package has {actual_size} bytes; expected {package.size_bytes}"
        )
    actual_digest = file_digest(partial)
    if actual_digest != package.sha256:
        partial.unlink(missing_ok=True)
        raise OfiqIntegrityError(
            f"downloaded OFIQ package has sha256 {actual_digest}; expected {package.sha256}"
        )
    os.replace(partial, destination)


def _wanted_member(name: str, platform_key: str) -> bool:
    file_names = {
        "OFIQ-Release/README.md",
        "OFIQ-Release/source/LICENSE.md",
        "OFIQ-Release/source/Version.txt",
        "OFIQ-Release/source/data/ofiq_config.jaxn",
    }
    prefixes = (
        f"OFIQ-Release/releases/{platform_key}/",
        "OFIQ-Release/source/data/models/",
        "OFIQ-Release/source/data/tests/images/",
        "OFIQ-Release/source/data/tests/expected_results/",
    )
    return name in file_names or any(name.startswith(prefix) for prefix in prefixes)


def _safe_destination(staging: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise OfiqIntegrityError(f"unsafe path in OFIQ package: {member_name}")
    return staging.joinpath(*relative.parts)


def _extract_package(
    archive_path: Path,
    destination_root: Path,
    package: OfiqPackageSpec,
    platform_key: str,
) -> OfiqInstallation:
    if destination_root.exists():
        message = (
            f"an incomplete OFIQ installation already exists at {destination_root}; "
            "remove it explicitly"
        )
        raise OfiqIntegrityError(message)

    destination_root.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ofiq-install-", dir=destination_root.parent.parent
    ) as raw:
        staging = Path(raw)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not _wanted_member(member.filename, platform_key):
                    continue
                target = _safe_destination(staging, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=_COPY_CHUNK_BYTES)

        staged_root = staging / "OFIQ-Release"
        installation = OfiqInstallation(staged_root, package, platform_key)
        installation.validate(require_marker=False)
        marker = {
            "schema_version": 1,
            "version": package.version,
            "source_revision": package.source_revision,
            "package_sha256": package.sha256,
        }
        (staged_root / _MARKER_NAME).write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_root, destination_root)

    installed = OfiqInstallation(destination_root, package, platform_key)
    installed.validate()
    return installed


def resolve_reference_ofiq(
    settings: Settings | None = None,
    *,
    allow_download: bool | None = None,
    package: OfiqPackageSpec = OFIQ_REFERENCE,
    platform_key: str | None = None,
) -> OfiqInstallation:
    """Return the verified reference installation, downloading it only when allowed."""
    resolved = settings or load_settings()
    selected_platform = platform_key or _platform_key()
    root = _install_root(resolved, package)
    installation = OfiqInstallation(root, package, selected_platform)
    if root.exists():
        installation.validate()
        return installation

    archive = _archive_path(resolved, package)
    may_download = resolved.allow_download if allow_download is None else allow_download
    if archive.is_file():
        if archive.stat().st_size != package.size_bytes:
            raise OfiqIntegrityError(f"cached OFIQ package at {archive} has the wrong size")
        actual_digest = file_digest(archive)
        if actual_digest != package.sha256:
            raise OfiqIntegrityError(
                f"cached OFIQ package at {archive} has sha256 {actual_digest}; "
                f"expected {package.sha256}"
            )
    elif may_download:
        _download_package(package, archive)
    else:
        raise OfiqNotAvailableError(
            f"OFIQ {package.version} is not installed under {resolved.ofiq_dir} and downloading "
            "is disabled; run `portraitkit ofiq fetch` while online"
        )

    return _extract_package(archive, root, package, selected_platform)


@dataclass(frozen=True, slots=True)
class OfiqMeasurement:
    """One native and mapped scalar ISO/IEC 29794-5 quality result."""

    name: str
    native_score: float
    scalar_score: float

    @property
    def assessed(self) -> bool:
        return self.scalar_score >= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_score": self.native_score,
            "scalar_score": self.scalar_score,
            "assessed": self.assessed,
        }


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    filename: str
    measurements: tuple[OfiqMeasurement, ...]


def _parse_number(value: str | None, *, column: str) -> float:
    if value is None or not value.strip():
        raise OfiqOutputError(f"OFIQ column {column!r} is empty")
    try:
        return float(value)
    except ValueError as error:
        raise OfiqOutputError(f"OFIQ column {column!r} is not numeric: {value!r}") from error


def parse_ofiq_csv(text: str) -> tuple[_ParsedRow, ...]:
    """Parse OFIQ 1.0.x or 1.1+ semicolon output without losing failure scores."""
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff")), delimiter=";"))
    if not rows or not rows[0] or rows[0][0] != "Filename":
        raise OfiqOutputError("OFIQ report does not start with a Filename column")
    header = [column for column in rows[0] if column]
    if len(header) != len(set(header)):
        raise OfiqOutputError("OFIQ report contains duplicate columns")

    parsed: list[_ParsedRow] = []
    for values in rows[1:]:
        if not values or not any(value.strip() for value in values):
            continue
        record = dict(zip(header, values, strict=False))
        filename = record.get("Filename", "").strip()
        if not filename:
            raise OfiqOutputError("OFIQ report contains a row without a filename")

        native: dict[str, float] = {}
        scalar: dict[str, float] = {}
        for column, value in record.items():
            if column == "Filename" or column in _METADATA_COLUMNS:
                continue
            if column.endswith(".scalar"):
                scalar[column.removesuffix(".scalar")] = _parse_number(value, column=column)
            else:
                name = column.removesuffix(".native")
                native[name] = _parse_number(value, column=column)

        if native.keys() != scalar.keys():
            missing_native = sorted(scalar.keys() - native.keys())
            missing_scalar = sorted(native.keys() - scalar.keys())
            raise OfiqOutputError(
                "OFIQ native/scalar columns disagree; "
                f"missing native={missing_native}, missing scalar={missing_scalar}"
            )
        measurements = tuple(OfiqMeasurement(name, native[name], scalar[name]) for name in native)
        parsed.append(_ParsedRow(filename, measurements))

    if not parsed:
        raise OfiqOutputError("OFIQ report contains no result rows")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class OfiqResult:
    """Quality measurements for one image plus exact scorer provenance."""

    image: Path
    measurements: tuple[OfiqMeasurement, ...]
    provenance: OfiqProvenance

    def measurement(self, name: str) -> OfiqMeasurement:
        for measurement in self.measurements:
            if measurement.name == name:
                return measurement
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": str(self.image),
            "measurements": {
                measurement.name: measurement.to_dict() for measurement in self.measurements
            },
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OfiqComparison:
    """Before/after result pair for a PortraitKit crop."""

    before: OfiqResult
    after: OfiqResult

    @property
    def scalar_deltas(self) -> Mapping[str, float]:
        before = {item.name: item.scalar_score for item in self.before.measurements}
        return {
            item.name: item.scalar_score - before[item.name]
            for item in self.after.measurements
            if item.name in before
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "scalar_deltas": dict(self.scalar_deltas),
        }


class OfiqScorer:
    """Invoke the official sample executable without a shell or in-process native code."""

    def __init__(self, installation: OfiqInstallation, *, timeout_seconds: float = 300.0):
        installation.validate()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.installation = installation
        self.timeout_seconds = timeout_seconds
        self._provenance: OfiqProvenance | None = None

    @property
    def provenance(self) -> OfiqProvenance:
        if self._provenance is None:
            self._provenance = self.installation.provenance()
        return self._provenance

    def score(self, input_path: Path) -> tuple[OfiqResult, ...]:
        """Score one image or every PNG/JPEG image in a directory."""
        source = Path(input_path)
        if not source.exists() or not (source.is_file() or source.is_dir()):
            raise OfiqNotAvailableError(f"OFIQ input does not exist: {source}")

        with tempfile.TemporaryDirectory(prefix="portraitkit-ofiq-") as raw:
            report = Path(raw) / "ofiq.csv"
            command = [
                str(self.installation.executable.resolve()),
                "-c",
                str(self.installation.config.resolve()),
                "-i",
                str(source.resolve()),
                "-o",
                str(report.resolve()),
            ]
            environment = os.environ.copy()
            if not self.installation.platform_key.startswith("win"):
                binary_dir = str(self.installation.executable.parent)
                current = environment.get("LD_LIBRARY_PATH", "")
                environment["LD_LIBRARY_PATH"] = (
                    binary_dir if not current else binary_dir + os.pathsep + current
                )
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.installation.executable.parent.resolve(),
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise OfiqExecutionError(f"OFIQ could not complete: {error}") from error
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-1000:]
                raise OfiqExecutionError(
                    f"OFIQ exited with code {completed.returncode}: {detail or 'no diagnostic'}"
                )
            if not report.is_file():
                raise OfiqOutputError("OFIQ exited successfully without producing its CSV report")

            parsed = parse_ofiq_csv(report.read_text(encoding="utf-8-sig"))

        provenance = self.provenance
        if source.is_file() and len(parsed) == 1:
            return (OfiqResult(source, parsed[0].measurements, provenance),)
        return tuple(OfiqResult(Path(row.filename), row.measurements, provenance) for row in parsed)

    def compare(self, before: np.ndarray, after: np.ndarray) -> OfiqComparison:
        """Score two RGB arrays in one OFIQ initialization and return their deltas."""
        with tempfile.TemporaryDirectory(prefix="portraitkit-ofiq-pair-") as raw:
            directory = Path(raw)
            for name, pixels in (("before.png", before), ("after.png", after)):
                array = np.asarray(pixels)
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError(f"{name} must be an HxWx3 uint8 RGB array")
                Image.fromarray(array).save(directory / name)
            results = self.score(directory)

        by_name = {result.image.name: result for result in results}
        try:
            return OfiqComparison(by_name["before.png"], by_name["after.png"])
        except KeyError as error:
            raise OfiqOutputError("OFIQ did not return both before.png and after.png") from error
