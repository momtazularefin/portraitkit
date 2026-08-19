"""BRIA AI RMBG-1.4 background removal adapter.

RMBG-1.4 operates on 1024x1024 RGB tensors normalized with standard ImageNet statistics
under a direct bilinear stretch resize mode.

Its output tensor is decoded into a 1-channel alpha matte in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import numpy as np

from portraitkit.errors import ModelError
from portraitkit.matting.base import MattingAdapter, register_matting_adapter
from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.types import ImageSize

__all__ = ["RMBG14_CONTRACT", "RMBG14Adapter", "decode_rmbg"]

INPUT_SIZE = ImageSize(width=1024, height=1024)

RMBG14_CONTRACT = PreprocessContract(
    input_name="input",
    input_size=INPUT_SIZE,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(123.675, 116.28, 103.53),
    scale=1.0 / 58.395,
    resize_mode=ResizeMode.BILINEAR_STRETCH,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def decode_rmbg(outputs: list[np.ndarray]) -> np.ndarray:
    """Decode raw RMBG output into a 2D alpha matte.

    Args:
        outputs: Output tensors returned by the RMBG session.

    Returns:
        ``(1024, 1024)`` float32 array in ``[0.0, 1.0]``.
    """
    if not outputs:
        msg = "RMBG expects at least 1 output tensor, got 0"
        raise ModelError(msg)

    raw = np.asarray(outputs[0], dtype=np.float32)
    matte = np.squeeze(raw)
    if matte.ndim != 2:
        msg = f"expected a 2D matte after squeezing, got shape {matte.shape}"
        raise ModelError(msg)

    if matte.size and (matte.min() < 0.0 or matte.max() > 1.0):
        matte = _sigmoid(matte)

    return np.clip(matte, 0.0, 1.0).astype(np.float32)


class RMBG14Adapter(MattingAdapter):
    """Adapter for the BRIA AI RMBG-1.4 background removal model."""

    @property
    def name(self) -> str:
        return "rmbg-1.4"

    @property
    def contract(self) -> PreprocessContract:
        return RMBG14_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return decode_rmbg(outputs)


register_matting_adapter("rmbg-1.4", RMBG14Adapter)
