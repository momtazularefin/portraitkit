"""DIS / IS-Net dichotomous image segmentation adapter.

IS-Net (Qin et al., ECCV 2022) is designed for accurate dichotomous image segmentation
and general background removal at high resolution (1024x1024).

Its output is decoded into a 1-channel alpha matte in ``[0.0, 1.0]``.
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

__all__ = ["ISNET_CONTRACT", "ISNetAdapter", "decode_isnet"]

INPUT_SIZE = ImageSize(width=1024, height=1024)

ISNET_CONTRACT = PreprocessContract(
    input_name="input.1",
    input_size=INPUT_SIZE,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(123.675, 116.28, 103.53),
    scale=1.0 / 58.395,
    resize_mode=ResizeMode.BILINEAR_STRETCH,
)


def decode_isnet(outputs: list[np.ndarray]) -> np.ndarray:
    """Decode raw IS-Net output into a 2D alpha matte.

    Args:
        outputs: Output tensors returned by the IS-Net session.

    Returns:
        ``(1024, 1024)`` float32 array in ``[0.0, 1.0]``.
    """
    if not outputs:
        msg = "IS-Net expects at least 1 output tensor, got 0"
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


class ISNetAdapter(MattingAdapter):
    """Adapter for the DIS / IS-Net dichotomous image segmentation artifact."""

    @property
    def name(self) -> str:
        return "isnet-general-use"

    @property
    def contract(self) -> PreprocessContract:
        return ISNET_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return decode_isnet(outputs)


register_matting_adapter("isnet-general-use", ISNetAdapter)
