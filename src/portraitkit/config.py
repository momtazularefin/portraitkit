"""Environment-based configuration.

PortraitKit reads every filesystem location and network toggle from the environment so
that nothing in the package or its tests depends on an absolute path (NFR5). Defaults
are relative to the current working directory, which keeps a fresh clone runnable with
no configuration at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from portraitkit.errors import ConfigError

__all__ = ["Settings", "load_settings"]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved runtime configuration."""

    model_dir: Path
    """Cache directory for downloaded model artifacts."""

    data_dir: Path
    """Root for public evaluation datasets."""

    output_dir: Path
    """Destination for pipeline output and benchmark reports."""

    allow_download: bool
    """Whether missing model artifacts may be fetched over the network."""

    def ensure_directories(self) -> None:
        """Create the configured directories if they do not exist."""
        for directory in (self.model_dir, self.data_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _read_path(env: Mapping[str, str], key: str, default: str) -> Path:
    raw = env.get(key, "").strip() or default
    return Path(raw).expanduser()


def _read_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    msg = f"{key} must be a boolean value, got {raw!r}"
    raise ConfigError(msg)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from ``env``, defaulting to :data:`os.environ`.

    Args:
        env: Mapping to read instead of the process environment. Tests pass an
            explicit mapping so they never mutate global state.

    Returns:
        The resolved settings. Directories are not created; call
        :meth:`Settings.ensure_directories` when they are actually needed.

    Raises:
        ConfigError: If a boolean-valued variable cannot be parsed.
    """
    source = os.environ if env is None else env
    return Settings(
        model_dir=_read_path(source, "PORTRAITKIT_MODEL_DIR", "./models"),
        data_dir=_read_path(source, "PORTRAITKIT_DATA_DIR", "./data"),
        output_dir=_read_path(source, "PORTRAITKIT_OUTPUT_DIR", "./output"),
        allow_download=_read_bool(source, "PORTRAITKIT_ALLOW_DOWNLOAD", default=True),
    )
