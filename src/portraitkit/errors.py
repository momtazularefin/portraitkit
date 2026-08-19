"""Exception hierarchy for PortraitKit.

Every error raised by the public API derives from :class:`PortraitKitError` so that
callers can trap the whole library with a single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "AnnotationError",
    "ConfigError",
    "ImageLoadError",
    "ModelError",
    "ModelIntegrityError",
    "ModelNotAvailableError",
    "OfiqError",
    "OfiqExecutionError",
    "OfiqIntegrityError",
    "OfiqNotAvailableError",
    "OfiqOutputError",
    "PortraitKitError",
    "StageError",
]


class PortraitKitError(Exception):
    """Base class for every PortraitKit error."""


class ConfigError(PortraitKitError):
    """Raised when configuration is missing, malformed, or contradictory."""


class AnnotationError(PortraitKitError):
    """Raised when an evaluation annotation set is missing, malformed, or inconsistent."""


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


class OfiqError(PortraitKitError):
    """Base class for failures at the optional external OFIQ boundary."""


class OfiqNotAvailableError(OfiqError):
    """Raised when the pinned OFIQ reference package is not available locally."""


class OfiqIntegrityError(OfiqError):
    """Raised when an OFIQ package or installation fails provenance checks."""


class OfiqExecutionError(OfiqError):
    """Raised when the external OFIQ process cannot complete successfully."""


class OfiqOutputError(OfiqError):
    """Raised when OFIQ emits a missing, malformed, or contradictory report."""
