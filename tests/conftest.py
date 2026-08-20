"""Shared test fixtures and synthetic image builders.

Tests use generated images rather than checked-in photographs. Nothing in this
repository may carry portrait data, and a positionally coded synthetic image proves
coordinate correctness far more precisely than a real face ever could.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from portraitkit.types import BoundingBox, FaceDetection, FaceLandmarks5

CODED_WIDTH = 7
CODED_HEIGHT = 4


def coded_array(width: int = CODED_WIDTH, height: int = CODED_HEIGHT) -> np.ndarray:
    """Build an image whose every pixel value encodes its own coordinates.

    The red channel encodes ``x``, the green channel encodes ``y``, and blue is
    constant. Every pixel is therefore unique, so a geometric transform can be verified
    by looking up where a specific value landed instead of eyeballing an image.
    """
    if width > 8 or height > 4:
        msg = "coded images stay small so channel values remain unambiguous"
        raise ValueError(msg)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixels[y, x] = (x * 30 + 5, y * 60 + 5, 200)
    return pixels


def coded_image(width: int = CODED_WIDTH, height: int = CODED_HEIGHT) -> Image.Image:
    """Return :func:`coded_array` as a Pillow RGB image."""
    return Image.fromarray(coded_array(width=width, height=height), mode="RGB")


def solid_image(width: int, height: int, color: tuple[int, int, int] = (128, 96, 64)) -> np.ndarray:
    """Return a uniformly colored ``(H, W, 3)`` uint8 array."""
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:, :] = color
    return pixels


def noise_image(width: int, height: int, seed: int = 7) -> np.ndarray:
    """Return a deterministic high-entropy image.

    Solid colors compress to almost nothing, which makes truncation tests degenerate.
    Seeded noise keeps the encoded payload large enough to truncate meaningfully while
    staying reproducible.
    """
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A small, structurally valid JPEG payload ending in a proper EOI marker."""
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(noise_image(64, 48), mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write_tiny_onnx(path: Path, *, dynamic_batch: bool = False) -> Path:
    """Write a minimal valid ONNX model for exercising the session factory.

    Building the model here rather than checking one in keeps the session tests free of
    binary fixtures and free of any network dependency, so CI covers the inference
    boundary without downloading weights.
    """
    import onnx
    from onnx import TensorProto, helper

    batch: object = "batch" if dynamic_batch else 1
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3])]
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "tiny", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


class StubDetector:
    """Stands in for a detector adapter so tests can run without model files."""

    def __init__(
        self,
        boxes: list[BoundingBox] | None = None,
        landmarks: list[FaceLandmarks5] | None = None,
        name: str = "stub-detector",
        config: object | None = None,
    ) -> None:
        from portraitkit.detection.base import DetectorConfig
        from portraitkit.models.session import SessionInfo
        from portraitkit.types import FaceDetection

        self.name = name
        self.config = config or DetectorConfig()
        self.info = SessionInfo(inputs=(), outputs=(), providers=("CPUExecutionProvider",))
        self._detections: list[FaceDetection] = []
        if boxes:
            for i, box in enumerate(boxes):
                lm = landmarks[i] if landmarks and i < len(landmarks) else None
                self._detections.append(FaceDetection(box=box, score=0.95, landmarks=lm))

    def detect(self, image_rgb: np.ndarray) -> tuple[FaceDetection, ...]:
        del image_rgb
        return tuple(self._detections)
