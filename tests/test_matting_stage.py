"""Unit tests for the MattingStage and foreground/background compositing."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.matting.base import MattingAdapter
from portraitkit.matting.stage import (
    MattingResult,
    MattingStage,
    MattingStageConfig,
    composite_matte,
    parse_color,
)
from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.types import ImageSize
from tests.conftest import solid_image


class DummyMatter(MattingAdapter):
    """Deterministic dummy matting adapter for testing."""

    def __init__(self, alpha_map: np.ndarray | None = None) -> None:
        self._alpha_map = alpha_map
        # We don't call super().__init__ to avoid needing an ONNX file on disk for unit testing

    @property
    def name(self) -> str:
        return "dummy-matter"

    @property
    def contract(self) -> PreprocessContract:
        return PreprocessContract(
            input_name="input",
            input_size=ImageSize(width=64, height=64),
            color_order=ColorOrder.RGB,
            layout=TensorLayout.NCHW,
            mean=(0.0, 0.0, 0.0),
            scale=1.0,
            resize_mode=ResizeMode.BILINEAR_STRETCH,
        )

    def decode(self, outputs: list[np.ndarray]) -> np.ndarray:
        return np.zeros((64, 64), dtype=np.float32)

    def predict_alpha(self, image_rgb: np.ndarray) -> np.ndarray:
        if self._alpha_map is not None:
            return self._alpha_map.copy()
        h, w = image_rgb.shape[:2]
        matte = np.zeros((h, w), dtype=np.float32)
        matte[:, : w // 2] = 1.0  # Left half foreground (1.0), right half background (0.0)
        return matte

    @property
    def info(self):
        from types import SimpleNamespace

        return SimpleNamespace(providers=["CPUExecutionProvider"])


def test_parse_color_named_and_hex() -> None:
    assert parse_color("white") == (255, 255, 255)
    assert parse_color("black") == (0, 0, 0)
    assert parse_color("transparent") is None
    assert parse_color("none") is None
    assert parse_color("#ff0000") == (255, 0, 0)
    assert parse_color("00ff00") == (0, 255, 0)
    assert parse_color("#fff") == (255, 255, 255)
    assert parse_color((10, 20, 30)) == (10, 20, 30)


def test_parse_color_invalid_raises() -> None:
    with pytest.raises(ValueError, match="invalid hex color"):
        parse_color("not-a-color")

    with pytest.raises(ValueError, match="RGB color components"):
        parse_color((300, 0, 0))  # type: ignore[arg-type]


def test_composite_matte_solid_white_and_transparency() -> None:
    # 10x10 green image (0, 255, 0)
    fg = np.zeros((10, 10, 3), dtype=np.uint8)
    fg[:, :, 1] = 255

    # Left half foreground (1.0), right half background (0.0)
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[:, :5] = 1.0

    rgb_out, rgba_out = composite_matte(fg, alpha, background_color=(255, 255, 255))

    # In rgb_out:
    # Left half should be foreground green (0, 255, 0)
    assert np.array_equal(rgb_out[:, :5], fg[:, :5])
    # Right half should be background white (255, 255, 255)
    assert np.all(rgb_out[:, 5:] == [255, 255, 255])

    # In rgba_out:
    # Alpha channel on left half should be 255, right half 0
    assert np.all(rgba_out[:, :5, 3] == 255)
    assert np.all(rgba_out[:, 5:, 3] == 0)


def test_composite_matte_threshold_binarizes() -> None:
    fg = np.full((4, 4, 3), 100, dtype=np.uint8)
    alpha = np.asarray(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=np.float32,
    )

    _, rgba_out = composite_matte(fg, alpha, threshold=0.5)

    assert np.all(rgba_out[:, :2, 3] == 0)
    assert np.all(rgba_out[:, 2:, 3] == 255)


def test_matting_stage_runs_end_to_end() -> None:
    matter = DummyMatter()
    config = MattingStageConfig(background_color=(255, 255, 255))
    stage = MattingStage(matter, config)

    image = solid_image(width=100, height=80, color=(100, 150, 200))
    result = stage.run(image)

    assert isinstance(result, MattingResult)
    assert result.ok
    assert result.alpha_matte.shape == (80, 100)
    assert result.image_rgb.shape == (80, 100, 3)
    assert result.image_rgba.shape == (80, 100, 4)
    assert result.image_size == ImageSize(width=100, height=80)
    assert result.matter == "dummy-matter"
    assert result.duration_ms >= 0.0
