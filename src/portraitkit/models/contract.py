"""Declared preprocessing contracts for model adapters.

Shared by every stage. Detection adapters and matting adapters differ in what they
decode, not in how an image becomes a tensor, so the contract lives with the model layer
rather than with either stage.

The legacy archive that preceded PortraitKit contained the same background-removal model
wrapped twice: one path stretched every input to a square and normalized with ImageNet
statistics, the other preserved aspect ratio and normalized to ``[-1, 1]``. Both wrappers
shipped. At most one could have been right, and nothing in either codebase could tell you
which, because preprocessing lived as loose statements scattered through the call site.

Here it is data instead. Every adapter declares a :class:`PreprocessContract`, the
contract builds the input tensor, and :meth:`PreprocessContract.validate_against` checks
the declaration against the signature ONNX Runtime actually reports. A mismatch between
what an adapter believes and what its artifact expects becomes a loud failure at load
time rather than a quietly wrong detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from portraitkit.errors import ModelError
from portraitkit.imaging.geometry import ResizeTransform, letterbox, stretch
from portraitkit.models.session import SessionInfo
from portraitkit.types import ImageSize

__all__ = ["ColorOrder", "PreprocessContract", "ResizeMode", "TensorLayout"]


class ColorOrder(StrEnum):
    """Channel order the model was trained to receive."""

    RGB = "rgb"
    BGR = "bgr"


class TensorLayout(StrEnum):
    """Axis order of the input tensor."""

    NCHW = "nchw"
    NHWC = "nhwc"


class ResizeMode(StrEnum):
    """How a source image is fitted to the model's input size."""

    LETTERBOX_TOP_LEFT = "letterbox_top_left"
    """Aspect-preserving scale, padded at the right and bottom. The convention the
    SCRFD and YuNet families are trained and evaluated with."""

    LETTERBOX_CENTER = "letterbox_center"
    """Aspect-preserving scale, padded symmetrically."""

    BILINEAR_STRETCH = "bilinear_stretch"
    """Direct anisotropic resize to input_size without padding. Stretches the image to fit
    the target canvas dimensions."""


@dataclass(frozen=True, slots=True)
class PreprocessContract:
    """A complete, executable description of one model's expected input.

    Attributes:
        input_name: Name of the ONNX input this tensor is bound to.
        input_size: Spatial size the model expects.
        color_order: Channel order to feed the model.
        layout: Axis order of the produced tensor.
        mean: Per-channel value subtracted before scaling.
        scale: Multiplier applied after mean subtraction.
        resize_mode: How the source image is fitted to ``input_size``.
    """

    input_name: str
    input_size: ImageSize
    color_order: ColorOrder
    layout: TensorLayout
    mean: tuple[float, float, float]
    scale: float
    resize_mode: ResizeMode = ResizeMode.LETTERBOX_TOP_LEFT

    def build_input(self, image_rgb: np.ndarray) -> tuple[np.ndarray, ResizeTransform]:
        """Turn an upright RGB image into this model's input tensor.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 array in RGB order, as produced by
                :func:`portraitkit.imaging.io.load_image`.

        Returns:
            The input tensor and the transform needed to map predictions back to
            source-image coordinates.
        """
        if self.resize_mode is ResizeMode.BILINEAR_STRETCH:
            canvas, transform = stretch(image_rgb, self.input_size)
        else:
            canvas, transform = letterbox(
                image_rgb,
                self.input_size,
                center=self.resize_mode is ResizeMode.LETTERBOX_CENTER,
            )

        if self.color_order is ColorOrder.BGR:
            canvas = canvas[:, :, ::-1]

        tensor = canvas.astype(np.float32)
        tensor -= np.asarray(self.mean, dtype=np.float32)
        tensor *= self.scale

        if self.layout is TensorLayout.NCHW:
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.ascontiguousarray(tensor[np.newaxis, ...]), transform

    def validate_against(self, info: SessionInfo) -> None:
        """Check this contract against the signature the loaded artifact reports.

        Raises:
            ModelError: If the input name is absent, or a fixed spatial dimension in the
                artifact disagrees with the declared input size.
        """
        spec = info.input_named(self.input_name)

        if len(spec.shape) != 4:
            msg = f"expected a 4-dimensional input for {self.input_name!r}, got shape {spec.shape}"
            raise ModelError(msg)

        if self.layout is TensorLayout.NCHW:
            channels, height, width = spec.shape[1], spec.shape[2], spec.shape[3]
        else:
            height, width, channels = spec.shape[1], spec.shape[2], spec.shape[3]

        if isinstance(channels, int) and channels != 3:
            msg = (
                f"contract declares three channels but {self.input_name!r} expects "
                f"{channels} under layout {self.layout.value}"
            )
            raise ModelError(msg)

        for label, declared, reported in (
            ("height", self.input_size.height, height),
            ("width", self.input_size.width, width),
        ):
            # A symbolic dimension means the artifact accepts any size, so only a fixed
            # dimension can contradict the declaration.
            if isinstance(reported, int) and reported != declared:
                msg = (
                    f"contract declares {label} {declared} for {self.input_name!r} but "
                    f"the model fixes it at {reported}"
                )
                raise ModelError(msg)
