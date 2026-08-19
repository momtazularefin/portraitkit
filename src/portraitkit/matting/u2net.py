"""U^2-Net human and salient object segmentation adapter.

U^2-Net (Qin et al., Pattern Recognition 2020) uses nested U-structures for multiscale
feature extraction. It operates on 320x320 RGB tensors with ImageNet normalization.

Its primary output tensor (d0) is decoded and min-max normalized into an alpha matte
in ``[0.0, 1.0]``.
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

__all__ = ["U2NET_CONTRACT", "U2NetAdapter", "U2NetPocketAdapter", "decode_u2net"]

INPUT_SIZE = ImageSize(width=320, height=320)

U2NET_CONTRACT = PreprocessContract(
    input_name="input.1",
    input_size=INPUT_SIZE,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(123.675, 116.28, 103.53),
    scale=1.0 / 58.395,
    resize_mode=ResizeMode.BILINEAR_STRETCH,
)


def decode_u2net(outputs: list[np.ndarray]) -> np.ndarray:
    """Decode raw U^2-Net output into a 2D alpha matte.

    Args:
        outputs: Output tensors returned by the U^2-Net session.

    Returns:
        ``(320, 320)`` float32 array in ``[0.0, 1.0]``.
    """
    if not outputs:
        msg = "U^2-Net expects at least 1 output tensor, got 0"
        raise ModelError(msg)

    raw = np.asarray(outputs[0], dtype=np.float32)
    matte = np.squeeze(raw)
    if matte.ndim != 2:
        msg = f"expected a 2D matte after squeezing, got shape {matte.shape}"
        raise ModelError(msg)

    min_val = float(matte.min())
    max_val = float(matte.max())
    if max_val > min_val:
        matte = (matte - min_val) / (max_val - min_val)

    return np.clip(matte, 0.0, 1.0).astype(np.float32)


class U2NetAdapter(MattingAdapter):
    """Adapter for the U^2-Net human segmentation artifact."""

    @property
    def name(self) -> str:
        return "u2net-human-seg"

    @property
    def contract(self) -> PreprocessContract:
        return U2NET_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return decode_u2net(outputs)


class U2NetPocketAdapter(MattingAdapter):
    """Adapter for the lightweight U^2-Net pocket (u2netp) artifact."""

    @property
    def name(self) -> str:
        return "u2netp"

    @property
    def contract(self) -> PreprocessContract:
        return U2NET_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return decode_u2net(outputs)


register_matting_adapter("u2net-human-seg", U2NetAdapter)
register_matting_adapter("u2netp", U2NetPocketAdapter)
