"""Unit tests for matting adapters, contracts, and decoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from portraitkit.errors import ModelError
from portraitkit.matting.base import MattingAdapter, build_matter
from portraitkit.matting.birefnet import BiRefNetAdapter, decode_birefnet
from portraitkit.matting.isnet import ISNetAdapter, decode_isnet
from portraitkit.matting.modnet import MODNetAdapter, decode_modnet
from portraitkit.matting.rmbg import RMBG14Adapter, decode_rmbg
from portraitkit.matting.u2net import U2NetAdapter, U2NetPocketAdapter, decode_u2net
from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.models.session import SessionInfo, TensorSpec
from portraitkit.types import ImageSize
from tests.conftest import solid_image

SIZE_512 = ImageSize(width=512, height=512)
SIZE_1024 = ImageSize(width=1024, height=1024)
SIZE_320 = ImageSize(width=320, height=320)


class StubMattingAdapter(MattingAdapter):
    """Stub adapter using a minimal ONNX session for testing."""

    def __init__(self, model_path: str | Path, contract: PreprocessContract) -> None:
        self._contract = contract
        super().__init__(model_path)

    @property
    def name(self) -> str:
        return "stub-matter"

    @property
    def contract(self) -> PreprocessContract:
        return self._contract

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return np.ones(
            (self.contract.input_size.height, self.contract.input_size.width), dtype=np.float32
        )


def make_info(shape: tuple[object, ...], name: str = "input") -> SessionInfo:
    return SessionInfo(
        inputs=(TensorSpec(name=name, shape=shape, dtype="tensor(float)"),),
        outputs=(),
        providers=("CPUExecutionProvider",),
    )


# --- decoder arithmetic -------------------------------------------------------------


def test_decode_modnet_squeezes_and_clips() -> None:
    raw = np.full((1, 1, 512, 512), 0.8, dtype=np.float32)
    raw[0, 0, 0, 0] = -0.1
    raw[0, 0, 0, 1] = 1.2

    matte = decode_modnet([raw])

    assert matte.shape == (512, 512)
    assert matte.dtype == np.float32
    assert matte[0, 0] == pytest.approx(0.0)
    assert matte[0, 1] == pytest.approx(1.0)
    assert matte[10, 10] == pytest.approx(0.8)


def test_decode_modnet_rejects_empty_outputs() -> None:
    with pytest.raises(ModelError, match="expects at least 1 output tensor"):
        decode_modnet([])


def test_decode_modnet_rejects_wrong_dimensions() -> None:
    with pytest.raises(ModelError, match="expected a 2D matte"):
        decode_modnet([np.ones((1, 512, 512, 3), dtype=np.float32)])


def test_decode_rmbg_applies_sigmoid_to_logits() -> None:
    # Logit 0.0 -> sigmoid 0.5, logit 20 -> ~1.0, logit -20 -> ~0.0
    raw = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
    raw[0, 0, 0, 0] = 20.0
    raw[0, 0, 0, 1] = -20.0

    matte = decode_rmbg([raw])

    assert matte.shape == (1024, 1024)
    assert matte[0, 0] == pytest.approx(1.0)
    assert matte[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert matte[10, 10] == pytest.approx(0.5)


def test_decode_rmbg_preserves_normalized_inputs() -> None:
    raw = np.full((1, 1, 1024, 1024), 0.7, dtype=np.float32)

    matte = decode_rmbg([raw])

    assert matte.shape == (1024, 1024)
    assert matte[0, 0] == pytest.approx(0.7)


def test_decode_rmbg_rejects_empty_outputs() -> None:
    with pytest.raises(ModelError, match="expects at least 1 output tensor"):
        decode_rmbg([])


def test_decode_u2net_normalizes_range() -> None:
    raw = np.full((1, 1, 320, 320), 2.0, dtype=np.float32)
    raw[0, 0, 0, 0] = 0.0
    raw[0, 0, 0, 1] = 4.0

    matte = decode_u2net([raw])

    assert matte.shape == (320, 320)
    assert matte[0, 0] == pytest.approx(0.0)
    assert matte[0, 1] == pytest.approx(1.0)
    assert matte[10, 10] == pytest.approx(0.5)


def test_decode_u2net_rejects_empty_outputs() -> None:
    with pytest.raises(ModelError, match="expects at least 1 output tensor"):
        decode_u2net([])


def test_decode_birefnet_applies_sigmoid() -> None:
    raw = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
    raw[0, 0, 0, 0] = 10.0
    raw[0, 0, 0, 1] = -10.0

    matte = decode_birefnet([raw])

    assert matte.shape == (1024, 1024)
    assert matte[0, 0] > 0.99
    assert matte[0, 1] < 0.01
    assert matte[10, 10] == pytest.approx(0.5)


def test_decode_birefnet_rejects_empty_outputs() -> None:
    with pytest.raises(ModelError, match="expects at least 1 output tensor"):
        decode_birefnet([])


def test_decode_isnet_normalizes_range() -> None:
    raw = np.full((1, 1, 1024, 1024), 5.0, dtype=np.float32)
    raw[0, 0, 0, 0] = 0.0
    raw[0, 0, 0, 1] = 10.0

    matte = decode_isnet([raw])

    assert matte.shape == (1024, 1024)
    assert matte[0, 0] == pytest.approx(0.0)
    assert matte[0, 1] == pytest.approx(1.0)
    assert matte[10, 10] == pytest.approx(0.5)


def test_decode_isnet_rejects_empty_outputs() -> None:
    with pytest.raises(ModelError, match="expects at least 1 output tensor"):
        decode_isnet([])


# --- contract validation & construction ----------------------------------------------


def test_modnet_adapter_contract() -> None:
    adapter_cls = MODNetAdapter
    assert adapter_cls.contract.fget(adapter_cls).input_size == SIZE_512
    assert adapter_cls.contract.fget(adapter_cls).resize_mode == ResizeMode.LETTERBOX_CENTER


def test_rmbg_adapter_contract() -> None:
    adapter_cls = RMBG14Adapter
    assert adapter_cls.contract.fget(adapter_cls).input_size == SIZE_1024
    assert adapter_cls.contract.fget(adapter_cls).resize_mode == ResizeMode.BILINEAR_STRETCH


def test_u2net_adapter_contracts() -> None:
    assert U2NetAdapter.contract.fget(U2NetAdapter).input_size == SIZE_320
    assert U2NetPocketAdapter.contract.fget(U2NetPocketAdapter).input_size == SIZE_320


def test_birefnet_adapter_contract() -> None:
    assert BiRefNetAdapter.contract.fget(BiRefNetAdapter).input_size == SIZE_1024


def test_isnet_adapter_contract() -> None:
    assert ISNetAdapter.contract.fget(ISNetAdapter).input_size == SIZE_1024


def test_build_matter_rejects_unknown_model() -> None:
    with pytest.raises(ModelError, match="no matting adapter for 'nonexistent'"):
        build_matter("nonexistent")


# --- end-to-end execution with tiny ONNX model ---------------------------------------


def test_adapter_predict_alpha_inverts_to_source_dimensions(tmp_path: Path) -> None:
    # Build a tiny ONNX model with input "x" of shape [1, 3, 64, 64]
    import onnx
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 64, 64])]
    # Identity node mapping x to y is not shape-compatible directly,
    # so use a slice node to extract 1 channel from the 3 input channels.
    node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])
    starts = helper.make_tensor("starts", TensorProto.INT64, [4], [0, 0, 0, 0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [4], [1, 1, 64, 64])
    axes = helper.make_tensor("axes", TensorProto.INT64, [4], [0, 1, 2, 3])
    steps = helper.make_tensor("steps", TensorProto.INT64, [4], [1, 1, 1, 1])

    graph = helper.make_graph(
        [node], "tiny_matter", inputs, outputs, initializer=[starts, ends, axes, steps]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    model_path = tmp_path / "tiny_matter.onnx"
    onnx.save(model, str(model_path))

    contract = PreprocessContract(
        input_name="x",
        input_size=ImageSize(width=64, height=64),
        color_order=ColorOrder.RGB,
        layout=TensorLayout.NCHW,
        mean=(0.0, 0.0, 0.0),
        scale=1.0,
        resize_mode=ResizeMode.BILINEAR_STRETCH,
    )

    adapter = StubMattingAdapter(model_path, contract)

    source_image = solid_image(width=120, height=80)
    alpha = adapter.predict_alpha(source_image)

    assert alpha.shape == (80, 120)
    assert alpha.dtype == np.float32
    assert np.allclose(alpha, 1.0)
