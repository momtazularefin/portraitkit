"""Declared preprocessing contracts."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.errors import ModelError
from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.models.session import SessionInfo, TensorSpec
from portraitkit.types import ImageSize
from tests.conftest import solid_image

SIZE = ImageSize(width=8, height=8)


def make_contract(**overrides: object) -> PreprocessContract:
    base: dict[str, object] = {
        "input_name": "input",
        "input_size": SIZE,
        "color_order": ColorOrder.RGB,
        "layout": TensorLayout.NCHW,
        "mean": (0.0, 0.0, 0.0),
        "scale": 1.0,
    }
    return PreprocessContract(**{**base, **overrides})  # type: ignore[arg-type]


def make_info(shape: tuple[object, ...], name: str = "input") -> SessionInfo:
    return SessionInfo(
        inputs=(TensorSpec(name=name, shape=shape, dtype="tensor(float)"),),
        outputs=(),
        providers=("CPUExecutionProvider",),
    )


def test_nchw_tensor_has_batch_and_channel_axes() -> None:
    tensor, _ = make_contract().build_input(solid_image(8, 8))

    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32


def test_nhwc_tensor_keeps_channels_last() -> None:
    tensor, _ = make_contract(layout=TensorLayout.NHWC).build_input(solid_image(8, 8))

    assert tensor.shape == (1, 8, 8, 3)


def test_rgb_contract_preserves_channel_order() -> None:
    image = solid_image(8, 8, color=(10, 20, 30))

    tensor, _ = make_contract().build_input(image)

    assert tensor[0, 0, 0, 0] == pytest.approx(10.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(30.0)


def test_bgr_contract_swaps_channel_order() -> None:
    image = solid_image(8, 8, color=(10, 20, 30))

    tensor, _ = make_contract(color_order=ColorOrder.BGR).build_input(image)

    assert tensor[0, 0, 0, 0] == pytest.approx(30.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(10.0)


def test_mean_and_scale_are_applied_in_order() -> None:
    image = solid_image(8, 8, color=(200, 200, 200))

    tensor, _ = make_contract(mean=(127.5, 127.5, 127.5), scale=1.0 / 128.0).build_input(image)

    assert tensor[0, 0, 0, 0] == pytest.approx((200.0 - 127.5) / 128.0)


def test_two_contracts_disagree_visibly_on_the_same_image() -> None:
    """The legacy failure this class exists to prevent.

    The archive shipped one model behind two wrappers that normalized differently. Both
    could not be right, and nothing recorded which was intended. Making preprocessing
    data means the disagreement is a value you can compare, not folklore.
    """
    image = solid_image(8, 8, color=(200, 200, 200))
    imagenet_style = make_contract(mean=(123.675, 116.28, 103.53), scale=1.0 / 58.395)
    symmetric = make_contract(mean=(127.5, 127.5, 127.5), scale=1.0 / 128.0)

    first, _ = imagenet_style.build_input(image)
    second, _ = symmetric.build_input(image)

    assert not np.allclose(first, second)


def test_transform_maps_predictions_back_to_source_coordinates() -> None:
    contract = make_contract(input_size=ImageSize(width=64, height=64))

    _, transform = contract.build_input(solid_image(width=128, height=64))

    assert transform.scale == pytest.approx(0.5)
    assert transform.source == ImageSize(width=128, height=64)


def test_centered_letterbox_pads_symmetrically() -> None:
    contract = make_contract(
        input_size=ImageSize(width=64, height=64), resize_mode=ResizeMode.LETTERBOX_CENTER
    )

    _, transform = contract.build_input(solid_image(width=128, height=64))

    assert transform.pad_y == pytest.approx(16.0)


def test_validation_accepts_a_matching_signature() -> None:
    make_contract().validate_against(make_info((1, 3, 8, 8)))


def test_validation_accepts_symbolic_spatial_dimensions() -> None:
    """A dynamic-input artifact accepts any size, so it cannot contradict the contract."""
    make_contract().validate_against(make_info((1, 3, "height", "width")))


def test_validation_rejects_an_unknown_input_name() -> None:
    with pytest.raises(ModelError, match="no input named 'input'"):
        make_contract().validate_against(make_info((1, 3, 8, 8), name="images"))


def test_validation_rejects_a_conflicting_fixed_size() -> None:
    with pytest.raises(ModelError, match=r"declares height 8 .* fixes it at 640"):
        make_contract().validate_against(make_info((1, 3, 640, 640)))


def test_validation_rejects_a_wrong_channel_count() -> None:
    with pytest.raises(ModelError, match="declares three channels"):
        make_contract().validate_against(make_info((1, 1, 8, 8)))


def test_validation_rejects_a_non_four_dimensional_input() -> None:
    with pytest.raises(ModelError, match="4-dimensional"):
        make_contract().validate_against(make_info((1, 3, 8)))


def test_layout_changes_which_axes_are_checked() -> None:
    """The same artifact shape means different things under NCHW and NHWC."""
    nhwc = make_contract(layout=TensorLayout.NHWC)

    nhwc.validate_against(make_info((1, 8, 8, 3)))
    with pytest.raises(ModelError, match="declares three channels"):
        nhwc.validate_against(make_info((1, 3, 8, 8)))


def test_stretch_mode_builds_stretched_tensor_without_padding() -> None:
    contract = make_contract(
        input_size=ImageSize(width=64, height=64),
        resize_mode=ResizeMode.BILINEAR_STRETCH,
    )

    tensor, transform = contract.build_input(solid_image(width=128, height=64))

    assert tensor.shape == (1, 3, 64, 64)
    assert transform.pad_x == 0.0
    assert transform.pad_y == 0.0
    assert transform.effective_scale_x == pytest.approx(0.5)
    assert transform.effective_scale_y == pytest.approx(1.0)
