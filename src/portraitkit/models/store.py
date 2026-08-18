"""Cache-aware resolution of registered model artifacts.

Resolution never trusts a file it has not verified. A cached artifact is checksummed
before use, and a download is verified in a temporary file and only then moved into
place, so an interrupted transfer can never be mistaken for a complete model.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

from portraitkit.config import Settings, load_settings
from portraitkit.errors import ModelIntegrityError, ModelNotAvailableError
from portraitkit.models.registry import ModelSpec, get_model

__all__ = [
    "cached_path",
    "file_digest",
    "is_cached",
    "resolve_model",
]

_CHUNK_BYTES: Final = 1 << 20
_DOWNLOAD_TIMEOUT_SECONDS: Final = 120
_USER_AGENT: Final = "portraitkit/0.1 (+https://github.com/momtazularefin/portraitkit)"


def file_digest(path: Path) -> str:
    """Return the lowercase hex SHA-256 of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _as_spec(model: str | ModelSpec) -> ModelSpec:
    return model if isinstance(model, ModelSpec) else get_model(model)


def cached_path(model: str | ModelSpec, settings: Settings | None = None) -> Path:
    """Return where ``model`` would live in the local cache, whether or not it exists."""
    spec = _as_spec(model)
    resolved = settings or load_settings()
    return resolved.model_dir / spec.filename


def is_cached(model: str | ModelSpec, settings: Settings | None = None) -> bool:
    """Whether ``model`` is already present locally with the expected size.

    This is a cheap presence check for callers deciding whether work is possible
    offline. It deliberately does not checksum the file; :func:`resolve_model` does that
    before the bytes are ever used.
    """
    spec = _as_spec(model)
    path = cached_path(spec, settings)
    return path.is_file() and path.stat().st_size == spec.size_bytes


def _download(spec: ModelSpec, destination: Path) -> None:
    """Fetch ``spec`` into ``destination``, verifying before installing.

    The transfer lands in a sibling ``.part`` file that is removed on any failure, so a
    failed download leaves the cache exactly as it was.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    # ModelSpec validates the https scheme at construction, so no other URL scheme
    # can reach the opener here.
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})

    try:
        with (
            urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as handle,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                handle.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        msg = f"could not download model {spec.name} from {spec.url}: {error}"
        raise ModelNotAvailableError(msg) from error

    actual = file_digest(partial)
    if actual != spec.sha256:
        partial.unlink(missing_ok=True)
        msg = (
            f"downloaded model {spec.name} failed verification: "
            f"expected sha256 {spec.sha256}, got {actual}"
        )
        raise ModelIntegrityError(msg)

    os.replace(partial, destination)


def resolve_model(
    model: str | ModelSpec,
    settings: Settings | None = None,
    *,
    allow_download: bool | None = None,
) -> Path:
    """Return a verified local path to ``model``, downloading it if permitted.

    Args:
        model: Registry name or an explicit :class:`ModelSpec`.
        settings: Configuration to use; defaults to the process environment.
        allow_download: Override the configured download policy for this call. Tests and
            offline runs pass ``False`` to guarantee no network access.

    Returns:
        Path to a cached file whose SHA-256 matches the registered digest.

    Raises:
        ModelNotAvailableError: The artifact is absent and may not be fetched, or the
            download failed.
        ModelIntegrityError: A cached or downloaded file does not match its digest.
    """
    spec = _as_spec(model)
    resolved = settings or load_settings()
    may_download = resolved.allow_download if allow_download is None else allow_download
    path = cached_path(spec, resolved)

    if path.is_file():
        actual = file_digest(path)
        if actual == spec.sha256:
            return path
        msg = (
            f"cached model {spec.name} at {path} does not match its registered digest: "
            f"expected {spec.sha256}, got {actual}. Delete the file to re-fetch it."
        )
        raise ModelIntegrityError(msg)

    if not may_download:
        msg = (
            f"model {spec.name} is not cached at {path} and downloading is disabled. "
            f"Fetch it from {spec.url} or set PORTRAITKIT_ALLOW_DOWNLOAD=1."
        )
        raise ModelNotAvailableError(msg)

    _download(spec, path)
    return path
