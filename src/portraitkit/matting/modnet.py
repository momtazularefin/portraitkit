"""MODNet photographic portrait matting adapter.

MODNet (Ke et al., AAAI 2022) is an objective-decomposed real-time portrait matting
architecture. It operates on 512x512 RGB tensors normalized symmetrically to ``[-1, 1]``
via mean 127.5 and scale 1/127.5.

Its single output tensor emits a 1-channel alpha matte with values in ``[0.0, 1.0]``.
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

__all__ = ["MODNET_CONTRACT", "MODNetAdapter", "decode_modnet"]

INPUT_SIZE = ImageSize(width=512, height=512)

MODNET_CONTRACT = PreprocessContract(
    input_name="input",
    input_size=INPUT_SIZE,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(127.5, 127.5, 127.5),
    scale=1.0 / 127.5,
    resize_mode=ResizeMode.LETTERBOX_CENTER,
)


def decode_modnet(outputs: list[np.ndarray]) -> np.ndarray:
    """Decode raw MODNet output into a 2D alpha matte.

    Args:
        outputs: Output tensors returned by the MODNet session.

    Returns:
        ``(H, W)`` float32 array in ``[0.0, 1.0]``.
    """
    if not outputs:
        msg = "MODNet expects at least 1 output tensor, got 0"
        raise ModelError(msg)

    raw = np.asarray(outputs[0], dtype=np.float32)
    matte = np.squeeze(raw)
    if matte.ndim != 2:
        msg = f"expected a 2D matte after squeezing, got shape {matte.shape}"
        raise ModelError(msg)

    return np.clip(matte, 0.0, 1.0).astype(np.float32)


class MODNetAdapter(MattingAdapter):
    """Adapter for the MODNet photographic portrait matting artifact."""

    @property
    def name(self) -> str:
        return "modnet-photographic"

    @property
    def contract(self) -> PreprocessContract:
        return MODNET_CONTRACT

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return decode_modnet(outputs)


register_matting_adapter("modnet-photographic", MODNetAdapter)
