"""Checksum-pinned public sample manifests for reproducible aggregate evaluations."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from portraitkit.config import Settings, load_settings
from portraitkit.errors import AnnotationError
from portraitkit.models.store import file_digest

__all__ = [
    "PublicSampleManifest",
    "PublicSampleSpec",
    "ResolvedPublicSamples",
    "load_public_sample_manifest",
    "resolve_public_samples",
]

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class PublicSampleSpec:
    """One synthetic or otherwise publication-safe public sample."""

    id: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicSampleManifest:
    """Pinned public sample selection and its licensing provenance."""

    name: str
    source: str
    source_revision: str
    license: str
    license_url: str
    selection: str
    samples: tuple[PublicSampleSpec, ...]


@dataclass(frozen=True, slots=True)
class ResolvedPublicSamples:
    """Verified local sample paths plus manifest metadata."""

    manifest: PublicSampleManifest
    paths: tuple[Path, ...]
    manifest_sha256: str


def _require_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AnnotationError(f"public sample manifest field {key!r} must be text")
    return value.strip()


def load_public_sample_manifest(path: Path) -> PublicSampleManifest:
    """Load and validate an identity-safe public sample manifest."""
    source_path = Path(path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError(
            f"cannot read public sample manifest {source_path}: {error}"
        ) from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise AnnotationError("public sample manifest schema_version must be 1")

    revision = _require_text(raw, "source_revision")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise AnnotationError("public sample source_revision must be a full Git SHA")
    sample_rows = raw.get("samples")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise AnnotationError("public sample manifest must contain samples")

    samples: list[PublicSampleSpec] = []
    seen_ids: set[str] = set()
    for row in sample_rows:
        if not isinstance(row, dict):
            raise AnnotationError("each public sample entry must be an object")
        sample_id = _require_text(row, "id")
        url = _require_text(row, "url")
        digest = _require_text(row, "sha256")
        size = row.get("size_bytes")
        if not _ID_PATTERN.fullmatch(sample_id) or sample_id in seen_ids:
            raise AnnotationError(f"public sample id is unsafe or duplicated: {sample_id!r}")
        if urlparse(url).scheme != "https" or revision not in url:
            raise AnnotationError(f"public sample URL must be HTTPS and pinned: {sample_id}")
        if not _SHA256_PATTERN.fullmatch(digest):
            raise AnnotationError(f"public sample SHA-256 is invalid: {sample_id}")
        if not isinstance(size, int) or size <= 0:
            raise AnnotationError(f"public sample size is invalid: {sample_id}")
        seen_ids.add(sample_id)
        samples.append(PublicSampleSpec(sample_id, url, size, digest))

    return PublicSampleManifest(
        name=_require_text(raw, "name"),
        source=_require_text(raw, "source"),
        source_revision=revision,
        license=_require_text(raw, "license"),
        license_url=_require_text(raw, "license_url"),
        selection=_require_text(raw, "selection"),
        samples=tuple(samples),
    )


def _sample_path(root: Path, sample: PublicSampleSpec) -> Path:
    suffix = Path(urlparse(sample.url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise AnnotationError(f"unsupported public sample image extension: {sample.id}")
    return root / f"{sample.id}{suffix}"


def _download(sample: PublicSampleSpec, target: Path) -> None:
    partial = target.with_name(f"{target.name}.part")
    request = urllib.request.Request(sample.url, headers={"User-Agent": "portraitkit/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1 << 20):
                output.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        raise AnnotationError(f"could not download public sample {sample.id}: {error}") from error
    if partial.stat().st_size != sample.size_bytes or file_digest(partial) != sample.sha256:
        partial.unlink(missing_ok=True)
        raise AnnotationError(f"downloaded public sample failed verification: {sample.id}")
    os.replace(partial, target)


def resolve_public_samples(
    manifest_path: Path,
    settings: Settings | None = None,
    *,
    allow_download: bool | None = None,
) -> ResolvedPublicSamples:
    """Resolve every manifest sample into the ignored data cache and verify its bytes."""
    manifest_file = Path(manifest_path)
    manifest = load_public_sample_manifest(manifest_file)
    resolved = settings or load_settings()
    may_download = resolved.allow_download if allow_download is None else allow_download
    root = resolved.data_dir / "public-samples" / manifest.name
    root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for sample in manifest.samples:
        target = _sample_path(root, sample)
        if not target.is_file():
            if not may_download:
                raise AnnotationError(
                    f"public sample {sample.id} is not cached and downloading is disabled"
                )
            _download(sample, target)
        if target.stat().st_size != sample.size_bytes or file_digest(target) != sample.sha256:
            raise AnnotationError(f"public sample failed its size or SHA-256 check: {sample.id}")
        paths.append(target)

    return ResolvedPublicSamples(
        manifest=manifest,
        paths=tuple(paths),
        manifest_sha256=file_digest(manifest_file),
    )
