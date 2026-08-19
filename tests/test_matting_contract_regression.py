"""Regression tests for the legacy preprocessing contradiction.

The legacy photo archive that preceded PortraitKit contained the same background-removal
model payload (MODNet) wrapped by two conflicting implementations:

1. ``RemoveBGUtil`` (.NET): Stretched every input to a 512x512 square, applied ImageNet
   normalization (mean [123.675, 116.28, 103.53], scale 1/58.395), and bound an input
   named ``img``.
2. ``inference_onnx.py`` (Python): Preserved aspect ratio, rounded dimensions to
   multiples of 32, normalized to [-1, 1] (mean 127.5, scale 1/127.5), and bound an input
   named ``input``.

Both shipped in production. Because preprocessing lived as loose statements scattered
through the code, nothing reported the mismatch.

These tests prove that:
- Declared contracts make the disagreement explicit and measurable values.
- Contract validation fails loudly at load time if an adapter targets the wrong signature.
- Applying square stretch to an aspect-ratio portrait creates measurable geometric
  distortion that the four benchmark metrics (SAD, MSE, Grad, Conn) immediately detect.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from portraitkit.errors import ModelError
from portraitkit.matting.metrics import matting_metrics
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

# The legacy .NET wrapper's declared preprocessing
LEGACY_DOTNET_CONTRACT = PreprocessContract(
    input_name="img",
    input_size=SIZE_512,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(123.675, 116.28, 103.53),
    scale=1.0 / 58.395,
    resize_mode=ResizeMode.BILINEAR_STRETCH,
)

# The legacy Python / MODNet reference preprocessing
LEGACY_PYTHON_CONTRACT = PreprocessContract(
    input_name="input",
    input_size=SIZE_512,
    color_order=ColorOrder.RGB,
    layout=TensorLayout.NCHW,
    mean=(127.5, 127.5, 127.5),
    scale=1.0 / 127.5,
    resize_mode=ResizeMode.LETTERBOX_CENTER,
)


def make_session_info(input_name: str, shape: tuple[object, ...]) -> SessionInfo:
    return SessionInfo(
        inputs=(TensorSpec(name=input_name, shape=shape, dtype="tensor(float)"),),
        outputs=(),
        providers=("CPUExecutionProvider",),
    )


def test_legacy_contracts_produce_fundamentally_different_tensors() -> None:
    """The two legacy preprocessing paths disagree on spatial geometry and values."""
    # A typical 3:4 portrait photo (300x400)
    image = solid_image(width=300, height=400, color=(180, 150, 120))

    dotnet_tensor, dotnet_transform = LEGACY_DOTNET_CONTRACT.build_input(image)
    python_tensor, python_transform = LEGACY_PYTHON_CONTRACT.build_input(image)

    # 1. Shapes are both 1x3x512x512, but dotnet stretched the content while python letterboxed
    assert dotnet_tensor.shape == (1, 3, 512, 512)
    assert python_tensor.shape == (1, 3, 512, 512)

    # 2. Python contract has padding columns on left and right
    assert python_transform.pad_x == pytest.approx(64.0)  # letterboxed horizontally
    assert python_transform.pad_y == 0.0
    assert dotnet_transform.pad_x == 0.0
    assert dotnet_transform.pad_y == 0.0

    # 3. Pixel normalization distributions differ
    # .NET ImageNet red channel: (180 - 123.675) / 58.395 = 0.9645
    # Python symmetric red channel: (180 - 127.5) / 127.5 = 0.4117
    assert dotnet_tensor[0, 0, 256, 256] == pytest.approx((180.0 - 123.675) / 58.395)
    assert python_tensor[0, 0, 256, 256] == pytest.approx((180.0 - 127.5) / 127.5)
    assert not np.allclose(dotnet_tensor, python_tensor)


def test_contract_validation_catches_input_name_contradiction() -> None:
    """An artifact with input 'input' rejects the .NET contract's 'img' at load time."""
    artifact_info = make_session_info("input", (1, 3, 512, 512))

    # Python contract succeeds
    LEGACY_PYTHON_CONTRACT.validate_against(artifact_info)

    # .NET contract fails with explicit error
    with pytest.raises(ModelError, match="no input named 'img'"):
        LEGACY_DOTNET_CONTRACT.validate_against(artifact_info)


def test_metrics_quantify_aspect_ratio_stretch_distortion() -> None:
    """Demonstrate that aspect-stretch vs letterbox creates severe benchmark error.

    Construct a ground-truth circle matte on a 2:1 portrait aspect image (width 100, height 200).
    Under letterbox preservation, the circle remains circular.
    Under square stretch, the circle becomes an ellipse, which the metrics heavily penalize.
    """
    height, width = 200, 100
    truth = np.zeros((height, width), dtype=np.float64)
    # Circle at center (50, 100) with radius 35
    cv2.circle(truth, (50, 100), 35, 1.0, -1)

    # 1. Aspect-preserving prediction: perfect circle
    aspect_preserving = truth.copy()
    metrics_preserving = matting_metrics(aspect_preserving, truth)
    assert metrics_preserving.sad == pytest.approx(0.0)
    assert metrics_preserving.mse == pytest.approx(0.0)

    # 2. Distorted prediction from square stretch without aspect correction:
    # Circle stretched into ellipse
    distorted = np.zeros((height, width), dtype=np.float64)
    cv2.ellipse(distorted, (50, 100), (35, 70), 0, 0, 360, 1.0, -1)

    metrics_distorted = matting_metrics(distorted, truth)

    # SAD, MSE, and gradient error must all increase significantly
    assert metrics_distorted.sad > 1.0  # Over 1000 pixels difference
    assert metrics_distorted.mse > 0.05
    assert metrics_distorted.gradient > 0.05
