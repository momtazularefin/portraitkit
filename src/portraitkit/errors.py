"""Exception hierarchy for PortraitKit.

Every error raised by the public API derives from :class:`PortraitKitError` so that
callers can trap the whole library with a single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "ImageLoadError",
    "ModelError",
    "ModelIntegrityError",
    "ModelNotAvailableError",
    "PortraitKitError",
    "StageError",
]


class PortraitKitError(Exception):
    """Base class for every PortraitKit error."""


class ConfigError(PortraitKitError):
    """Raised when configuration is missing, malformed, or contradictory."""


class ImageLoadError(PortraitKitError):
    """Raised when an image cannot be read, decoded, or converted to RGB."""


class StageError(PortraitKitError):
    """Raised when a pipeline stage cannot produce a result at all.

    A stage that runs successfully but finds nothing usable reports that through its
    typed result status instead of raising.
    """


class ModelError(PortraitKitError):
    """Base class for model resolution and inference errors."""


class ModelNotAvailableError(ModelError):
    """Raised when a required model artifact is absent and cannot be fetched."""


class ModelIntegrityError(ModelError):
    """Raised when a model artifact fails checksum verification."""
