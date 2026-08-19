"""Public evaluation samples are immutable, licensed, and cache-only in Git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from portraitkit.config import Settings
from portraitkit.errors import AnnotationError
from portraitkit.eval.samples import load_public_sample_manifest, resolve_public_samples


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        ofiq_dir=tmp_path / "models" / "ofiq",
        allow_download=False,
    )


def write_manifest(tmp_path: Path, payload: bytes = b"public synthetic fixture") -> Path:
    revision = "1" * 40
    manifest = {
        "schema_version": 1,
        "name": "synthetic-fixture",
        "source": "https://example.invalid/source",
        "source_revision": revision,
        "license": "CC0-1.0",
        "license_url": "https://example.invalid/license",
        "selection": "one generated fixture",
        "samples": [
            {
                "id": "sample-001",
                "url": f"https://example.invalid/{revision}/fixture.jpg",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_resolver_uses_a_verified_ignored_cache(tmp_path: Path) -> None:
    payload = b"public synthetic fixture"
    manifest_path = write_manifest(tmp_path, payload)
    settings = make_settings(tmp_path)
    cached = settings.data_dir / "public-samples" / "synthetic-fixture" / "sample-001.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload)

    resolved = resolve_public_samples(manifest_path, settings, allow_download=False)

    assert resolved.paths == (cached,)
    assert resolved.manifest.license == "CC0-1.0"
    assert len(resolved.manifest_sha256) == 64


def test_manifest_rejects_a_floating_sample_url(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["url"] = "https://example.invalid/main/fixture.jpg"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnnotationError, match="pinned"):
        load_public_sample_manifest(path)


def test_resolver_rejects_a_corrupt_cached_sample(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    settings = make_settings(tmp_path)
    cached = settings.data_dir / "public-samples" / "synthetic-fixture" / "sample-001.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"wrong")

    with pytest.raises(AnnotationError, match="size or SHA-256"):
        resolve_public_samples(manifest_path, settings, allow_download=False)
