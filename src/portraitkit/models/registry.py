"""The catalog of model artifacts PortraitKit knows how to fetch.

Every entry is pinned to an immutable upstream revision and a SHA-256 digest, so a
benchmark run months from now resolves the same bytes it resolved today. Reproducibility
is a benchmark's whole claim to authority; a floating ``main`` reference would quietly
invalidate it.

Each entry also records its license and whether that license permits commercial use.
PortraitKit itself is MIT, but model weights carry their own terms, and several
widely-used face detectors are released for non-commercial research only. Recording that
in the registry keeps the distinction visible to integrators instead of burying it in a
README somewhere upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from portraitkit.errors import ModelError

__all__ = [
    "DEFAULT_DETECTOR",
    "MODELS",
    "ModelSpec",
    "get_model",
    "model_names",
]

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A pinned, verifiable model artifact."""

    name: str
    """Registry key, also the CLI-facing identifier."""

    filename: str
    """Filename used inside the local model cache."""

    url: str
    """Direct download URL pinned to an immutable upstream revision."""

    sha256: str
    """Lowercase hex digest of the expected file contents."""

    size_bytes: int
    """Expected file size, used for a cheap sanity check and for progress reporting."""

    license: str
    """Short license name as published by the upstream distribution."""

    license_url: str
    """Where the license statement above was read from."""

    permits_commercial_use: bool
    """Whether the upstream license allows commercial use of the weights."""

    upstream: str
    """Human-readable provenance of the artifact."""

    notes: str = ""
    """Anything an integrator should know before selecting this model."""

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.match(self.sha256):
            msg = f"{self.name}: sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            raise ValueError(msg)
        if not self.url.startswith("https://"):
            msg = f"{self.name}: model URLs must use https, got {self.url!r}"
            raise ValueError(msg)
        if self.size_bytes <= 0:
            msg = f"{self.name}: size_bytes must be positive, got {self.size_bytes}"
            raise ValueError(msg)


_ENTRIES: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="yunet-2023mar",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "f12e12798e8314f7c074a6656816c048dcc95b7a/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size_bytes=232589,
        license="MIT",
        license_url=(
            "https://github.com/opencv/opencv_zoo/blob/"
            "f12e12798e8314f7c074a6656816c048dcc95b7a/"
            "models/face_detection_yunet/LICENSE"
        ),
        permits_commercial_use=True,
        upstream="OpenCV Zoo, from Shiqi Yu's libfacedetection lineage",
        notes=(
            "PortraitKit's default detector. 227 KiB, CPU-friendly, five-point landmarks, "
            "and an MIT license that imposes no downstream restriction on integrators."
        ),
    ),
    ModelSpec(
        name="scrfd-10g-bnkps",
        filename="scrfd_10g_bnkps.onnx",
        url=(
            "https://huggingface.co/immich-app/buffalo_l/resolve/"
            "d09715916a0778919a770c343533641e250b8699/detection/model.onnx"
        ),
        sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        size_bytes=16923827,
        license="Non-commercial research use only",
        license_url="https://github.com/deepinsight/insightface#license",
        permits_commercial_use=False,
        upstream="InsightFace buffalo_l detection model, mirrored by immich-app",
        notes=(
            "Stronger detector, opt-in only. InsightFace releases its trained models for "
            "non-commercial research purposes; the weights are therefore unsuitable as a "
            "default for a library integrators may ship commercially. Useful as a "
            "PortraitBench entrant and as a quality reference."
        ),
    ),
)

MODELS: Final[MappingProxyType[str, ModelSpec]] = MappingProxyType(
    {entry.name: entry for entry in _ENTRIES}
)
"""Read-only registry keyed by model name."""

DEFAULT_DETECTOR: Final = "yunet-2023mar"
"""The detector used when a caller expresses no preference.

Chosen for license clarity first and footprint second: an MIT-licensed default keeps
PortraitKit's own MIT promise meaningful all the way down to the weights.
"""


def model_names() -> tuple[str, ...]:
    """Return every registered model name, in registration order."""
    return tuple(MODELS)


def get_model(name: str) -> ModelSpec:
    """Look up a model specification by name.

    Raises:
        ModelError: If ``name`` is not registered.
    """
    try:
        return MODELS[name]
    except KeyError:
        known = ", ".join(model_names())
        msg = f"unknown model {name!r}; registered models are: {known}"
        raise ModelError(msg) from None
