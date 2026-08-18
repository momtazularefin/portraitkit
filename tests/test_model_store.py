"""Cache-aware model resolution.

Every test here runs offline. Network access is simulated by substituting the URL
opener, which keeps CI deterministic and proves the failure paths that a real download
would only exercise by accident.
"""

from __future__ import annotations

import hashlib
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from portraitkit.config import Settings
from portraitkit.errors import ModelIntegrityError, ModelNotAvailableError
from portraitkit.models import store
from portraitkit.models.registry import ModelSpec

PAYLOAD = b"pretend-onnx-bytes" * 16
PAYLOAD_DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def make_spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "name": "probe",
        "filename": "probe.onnx",
        "url": "https://example.invalid/probe.onnx",
        "sha256": PAYLOAD_DIGEST,
        "size_bytes": len(PAYLOAD),
        "license": "MIT",
        "license_url": "https://example.invalid/LICENSE",
        "permits_commercial_use": True,
        "upstream": "test fixture",
    }
    return ModelSpec(**{**base, **overrides})


def make_settings(tmp_path: Path, *, allow_download: bool = False) -> Settings:
    return Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        allow_download=allow_download,
    )


class FakeResponse:
    """A minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    """Install a fake URL opener and report how many times it was called."""
    calls: list[str] = []

    def install(payload: bytes | Exception) -> list[str]:
        def fake_urlopen(request: Any, timeout: float | None = None) -> FakeResponse:
            calls.append(getattr(request, "full_url", str(request)))
            if isinstance(payload, Exception):
                raise payload
            return FakeResponse(payload)

        monkeypatch.setattr(store.urllib.request, "urlopen", fake_urlopen)
        return calls

    return install


def test_file_digest_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(PAYLOAD)

    assert store.file_digest(path) == PAYLOAD_DIGEST


def test_cached_path_lives_under_the_configured_model_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    path = store.cached_path(make_spec(), settings)

    assert path == settings.model_dir / "probe.onnx"


def test_is_cached_requires_the_expected_size(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    spec = make_spec()
    settings.model_dir.mkdir(parents=True)
    target = settings.model_dir / spec.filename

    assert not store.is_cached(spec, settings)

    target.write_bytes(b"short")
    assert not store.is_cached(spec, settings)

    target.write_bytes(PAYLOAD)
    assert store.is_cached(spec, settings)


def test_resolve_returns_a_verified_cached_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    spec = make_spec()
    settings.model_dir.mkdir(parents=True)
    (settings.model_dir / spec.filename).write_bytes(PAYLOAD)

    resolved = store.resolve_model(spec, settings)

    assert resolved.read_bytes() == PAYLOAD


def test_resolve_rejects_a_corrupted_cached_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    spec = make_spec()
    settings.model_dir.mkdir(parents=True)
    (settings.model_dir / spec.filename).write_bytes(b"tampered payload of equal-ish size")

    with pytest.raises(ModelIntegrityError, match="does not match its registered digest"):
        store.resolve_model(spec, settings)


def test_resolve_refuses_to_download_when_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, allow_download=False)

    with pytest.raises(ModelNotAvailableError, match="downloading is disabled"):
        store.resolve_model(make_spec(), settings)


def test_resolve_downloads_and_installs(tmp_path: Path, serve) -> None:
    settings = make_settings(tmp_path, allow_download=True)
    spec = make_spec()
    calls = serve(PAYLOAD)

    resolved = store.resolve_model(spec, settings)

    assert resolved.read_bytes() == PAYLOAD
    assert calls == [spec.url]
    assert not list(settings.model_dir.glob("*.part"))


def test_a_second_resolve_uses_the_cache(tmp_path: Path, serve) -> None:
    settings = make_settings(tmp_path, allow_download=True)
    spec = make_spec()
    calls = serve(PAYLOAD)

    store.resolve_model(spec, settings)
    store.resolve_model(spec, settings)

    assert len(calls) == 1


def test_corrupted_download_is_rejected_and_leaves_no_artifact(tmp_path: Path, serve) -> None:
    """A truncated or tampered transfer must never be installed as a usable model."""
    settings = make_settings(tmp_path, allow_download=True)
    spec = make_spec()
    serve(b"wrong bytes entirely")

    with pytest.raises(ModelIntegrityError, match="failed verification"):
        store.resolve_model(spec, settings)

    assert not (settings.model_dir / spec.filename).exists()
    assert not list(settings.model_dir.glob("*.part"))


def test_network_failure_leaves_no_partial_file(tmp_path: Path, serve) -> None:
    settings = make_settings(tmp_path, allow_download=True)
    spec = make_spec()
    serve(urllib.error.URLError("connection refused"))

    with pytest.raises(ModelNotAvailableError, match="could not download model probe"):
        store.resolve_model(spec, settings)

    assert not (settings.model_dir / spec.filename).exists()
    assert not list(settings.model_dir.glob("*.part"))


def test_allow_download_argument_overrides_settings(tmp_path: Path, serve) -> None:
    settings = make_settings(tmp_path, allow_download=True)
    serve(PAYLOAD)

    with pytest.raises(ModelNotAvailableError):
        store.resolve_model(make_spec(), settings, allow_download=False)


def test_resolve_accepts_a_registry_name(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, allow_download=False)

    with pytest.raises(ModelNotAvailableError, match="yunet-2023mar"):
        store.resolve_model("yunet-2023mar", settings)
