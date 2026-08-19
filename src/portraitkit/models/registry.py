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
    "DEFAULT_MATTER",
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
    ModelSpec(
        name="modnet-photographic",
        filename="modnet_photographic.onnx",
        url=(
            "https://huggingface.co/Xenova/modnet/resolve/"
            "fa2fa546052fba4c08921230a26cc69a333fca12/onnx/model.onnx"
        ),
        sha256="07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
        size_bytes=25888640,
        license="Apache-2.0",
        license_url="https://github.com/ZHKKKe/MODNet/blob/master/LICENSE",
        permits_commercial_use=True,
        upstream="MODNet photographic portrait matting (Ke et al., AAAI 2022 / Xenova)",
        notes=(
            "PortraitKit's default matting model. 24.7 MiB, Apache-2.0 licensed, designed "
            "specifically for trimap-free portrait matting."
        ),
    ),
    ModelSpec(
        name="rmbg-1.4",
        filename="rmbg_1.4.onnx",
        url=(
            "https://huggingface.co/briaai/RMBG-1.4/resolve/"
            "2ceba5a5efaec153162aedea169f76caf9b46cf8/onnx/model.onnx"
        ),
        sha256="8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f",
        size_bytes=176153355,
        license="bria-rmbg-1.4 (Non-commercial research use only)",
        license_url="https://huggingface.co/briaai/RMBG-1.4",
        permits_commercial_use=False,
        upstream="BRIA AI RMBG-1.4 background removal",
        notes=(
            "High-quality background removal, opt-in only. BRIA releases RMBG-1.4 under a "
            "non-commercial license; commercial deployment requires an enterprise license "
            "from BRIA AI."
        ),
    ),
    ModelSpec(
        name="u2net-human-seg",
        filename="u2net_human_seg.onnx",
        url=(
            "https://huggingface.co/davidfant/rembg-u2net/resolve/"
            "82951e32735e4e6cd7bbf3e51279960d1ca3ef56/u2net_human_seg.onnx"
        ),
        sha256="01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
        size_bytes=175997641,
        license="Apache-2.0",
        license_url="https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE",
        permits_commercial_use=True,
        upstream="U^2-Net human segmentation (Qin et al., 2020 / rembg)",
        notes=("Salient human segmentation model. Apache-2.0 license permits commercial use."),
    ),
    ModelSpec(
        name="u2netp",
        filename="u2netp.onnx",
        url=(
            "https://huggingface.co/tomjackson2023/rembg/resolve/"
            "cd3a3d6767a7859efea31ef0f2f373582cf06d82/u2netp.onnx"
        ),
        sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        size_bytes=4574861,
        license="Apache-2.0",
        license_url="https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE",
        permits_commercial_use=True,
        upstream="U^2-Net pocket / lightweight variant (Qin et al., 2020 / rembg)",
        notes=("Lightweight 4.4 MiB U^2-Net variant for fast inference on CPU environments."),
    ),
    ModelSpec(
        name="birefnet-general",
        filename="birefnet_general.onnx",
        url=(
            "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/"
            "534d3c82d3bb8b2f0867db6dfbc3a525b8e42f67/onnx/model.onnx"
        ),
        sha256="58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        size_bytes=972666916,
        license="Apache-2.0",
        license_url="https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        permits_commercial_use=True,
        upstream="BiRefNet bilateral reference network (Zheng et al., 2024 / onnx-community)",
        notes=("High-resolution dichotomous image segmentation model. Apache-2.0 license."),
    ),
    ModelSpec(
        name="isnet-general-use",
        filename="isnet_general_use.onnx",
        url=(
            "https://huggingface.co/tomjackson2023/rembg/resolve/"
            "cd3a3d6767a7859efea31ef0f2f373582cf06d82/isnet-general-use.onnx"
        ),
        sha256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        size_bytes=178648008,
        license="Apache-2.0",
        license_url="https://github.com/xuebinqin/DIS/blob/main/LICENSE",
        permits_commercial_use=True,
        upstream="DIS / IS-Net (Qin et al., ECCV 2022 / rembg)",
        notes=("Dichotomous image segmentation model for general background removal."),
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

DEFAULT_MATTER: Final = "modnet-photographic"
"""The matting model used when a caller expresses no preference.

Chosen for license clarity (Apache-2.0), modest footprint (24.7 MiB), and domain alignment
with portrait matting.
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
