"""The background matting adapter interface.

Every matting model PortraitKit supports, and every model PortraitBench evaluates, is
reached through this interface. Adapters declare a preprocessing contract and implement
model-specific decoding of raw ONNX tensors; tensor preparation, execution, session
validation, and coordinate/aspect inversion back to source image space are shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from portraitkit.config import Settings
from portraitkit.errors import ModelError
from portraitkit.models.contract import PreprocessContract
from portraitkit.models.registry import DEFAULT_MATTER, get_model
from portraitkit.models.session import CPU_PROVIDER, create_session, describe_session
from portraitkit.models.store import resolve_model

if TYPE_CHECKING:
    from portraitkit.imaging.geometry import ResizeTransform

__all__ = ["MattingAdapter", "build_matter", "register_matting_adapter"]

_ADAPTERS: dict[str, type[MattingAdapter]] = {}


class MattingAdapter(ABC):
    """Base class for ONNX background-removal and matting adapters.

    Subclasses declare a preprocessing contract and implement :meth:`decode`. The
    template method :meth:`predict_alpha` owns preprocessing, inference, and inversion
    back to the original image dimensions.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: tuple[str, ...] = (CPU_PROVIDER,),
    ) -> None:
        self.session = create_session(model_path, providers=providers)
        self.info = describe_session(self.session)
        # Fail at construction if the artifact contradicts the declared contract, rather
        # than producing corrupt or silent failures at inference time.
        self.contract.validate_against(self.info)

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name of the model this adapter drives."""

    @property
    @abstractmethod
    def contract(self) -> PreprocessContract:
        """The preprocessing this adapter declares."""

    @abstractmethod
    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Turn raw session output into a 2D alpha matte in canvas dimensions.

        Args:
            outputs: Raw tensors returned by ONNX Runtime for this model.

        Returns:
            A 2D float array in ``[0.0, 1.0]`` with dimensions matching the canvas
            size declared in :attr:`contract`.
        """

    def predict_alpha(self, image_rgb: np.ndarray) -> np.ndarray:
        """Predict the alpha matte for an upright RGB image.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 array in RGB order.

        Returns:
            ``(H, W)`` float32 alpha matte in ``[0.0, 1.0]`` matching the input dimensions.
        """
        tensor, transform = self.contract.build_input(image_rgb)
        outputs = self.session.run(None, {self.contract.input_name: tensor})
        raw_matte = self.decode(list(outputs))
        return self._postprocess(raw_matte, transform)

    def _postprocess(self, raw_matte: np.ndarray, transform: ResizeTransform) -> np.ndarray:
        """Invert canvas padding/scaling back to source-image coordinates."""
        return transform.invert_matte(raw_matte)


def register_matting_adapter(name: str, adapter_cls: type[MattingAdapter]) -> None:
    """Register an adapter class for a model name."""
    _ADAPTERS[name] = adapter_cls


def build_matter(
    model: str = DEFAULT_MATTER,
    *,
    settings: Settings | None = None,
    model_path: str | Path | None = None,
    providers: tuple[str, ...] = (CPU_PROVIDER,),
    allow_download: bool | None = None,
) -> MattingAdapter:
    """Construct the matting adapter for ``model``, resolving its artifact if needed.

    Args:
        model: Registry name of the matting model.
        settings: Configuration used to locate the model cache.
        model_path: Bypass resolution and load this file directly.
        providers: Execution providers, CPU by default.
        allow_download: Override the configured download policy.

    Raises:
        ModelError: If no adapter is registered for ``model``.
    """
    try:
        adapter = _ADAPTERS[model]
    except KeyError:
        known = ", ".join(sorted(_ADAPTERS))
        msg = f"no matting adapter for {model!r}; available adapters are: {known}"
        raise ModelError(msg) from None

    if model_path is None:
        spec = get_model(model)
        model_path = resolve_model(spec, settings, allow_download=allow_download)
    return adapter(model_path, providers=providers)
