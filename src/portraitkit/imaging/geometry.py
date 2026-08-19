"""Resize transforms that stay invertible.

Detectors work at a fixed input size, so every image is rescaled before inference and
every prediction has to travel back to the coordinate space of the original picture. The
legacy archive that preceded this project lost track of that mapping in several places,
which is why the transform here is an explicit object with an inverse rather than a pair
of loose scale factors applied by hand at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from portraitkit.types import BoundingBox, ImageSize, Point

__all__ = ["ResizeTransform", "letterbox", "stretch"]


@dataclass(frozen=True, slots=True)
class ResizeTransform:
    """An aspect-preserving or stretched resize into a canvas.

    Forward direction maps source-image coordinates to canvas coordinates:
    ``canvas = source * scale + pad``. :meth:`invert_point`, :meth:`invert_box`,
    :meth:`invert_array`, and :meth:`invert_matte` undo it.
    """

    source: ImageSize
    canvas: ImageSize
    scale: float
    pad_x: float
    pad_y: float
    scale_x: float | None = None
    scale_y: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            msg = f"resize scale must be positive, got {self.scale}"
            raise ValueError(msg)
        if self.scale_x is not None and self.scale_x <= 0.0:
            msg = f"scale_x must be positive, got {self.scale_x}"
            raise ValueError(msg)
        if self.scale_y is not None and self.scale_y <= 0.0:
            msg = f"scale_y must be positive, got {self.scale_y}"
            raise ValueError(msg)

    @property
    def effective_scale_x(self) -> float:
        return self.scale_x if self.scale_x is not None else self.scale

    @property
    def effective_scale_y(self) -> float:
        return self.scale_y if self.scale_y is not None else self.scale

    def apply_point(self, point: Point) -> Point:
        """Map a source-image point onto the padded canvas."""
        return Point(
            x=point.x * self.effective_scale_x + self.pad_x,
            y=point.y * self.effective_scale_y + self.pad_y,
        )

    def invert_point(self, point: Point) -> Point:
        """Map a canvas point back to source-image coordinates."""
        return Point(
            x=(point.x - self.pad_x) / self.effective_scale_x,
            y=(point.y - self.pad_y) / self.effective_scale_y,
        )

    def invert_box(self, box: BoundingBox) -> BoundingBox:
        """Map a canvas box back to source-image coordinates."""
        top_left = self.invert_point(Point(x=box.x1, y=box.y1))
        bottom_right = self.invert_point(Point(x=box.x2, y=box.y2))
        return BoundingBox(x1=top_left.x, y1=top_left.y, x2=bottom_right.x, y2=bottom_right.y)

    def invert_array(self, points: np.ndarray) -> np.ndarray:
        """Map an ``(..., 2)`` array of canvas coordinates back to source coordinates."""
        offset = np.asarray([self.pad_x, self.pad_y], dtype=np.float32)
        scale_vec = np.asarray([self.effective_scale_x, self.effective_scale_y], dtype=np.float32)
        return (np.asarray(points, dtype=np.float32) - offset) / scale_vec

    def invert_matte(self, matte: np.ndarray) -> np.ndarray:
        """Map a canvas-sized 2D alpha matte back to source-image dimensions."""
        if matte.ndim != 2:
            msg = f"expected a 2D matte array, got shape {matte.shape}"
            raise ValueError(msg)
        if (matte.shape[0], matte.shape[1]) != (self.canvas.height, self.canvas.width):
            msg = (
                f"matte shape {matte.shape} does not match canvas dimensions "
                f"({self.canvas.height}, {self.canvas.width})"
            )
            raise ValueError(msg)

        pad_x = round(self.pad_x)
        pad_y = round(self.pad_y)
        scaled_w = max(1, round(self.source.width * self.effective_scale_x))
        scaled_h = max(1, round(self.source.height * self.effective_scale_y))

        active = matte[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w]
        interpolation = (
            cv2.INTER_AREA
            if (scaled_w > self.source.width or scaled_h > self.source.height)
            else cv2.INTER_LINEAR
        )
        restored = cv2.resize(
            active, (self.source.width, self.source.height), interpolation=interpolation
        )
        return np.clip(restored, 0.0, 1.0).astype(np.float32)


def letterbox(
    image: np.ndarray, target: ImageSize, *, center: bool = False
) -> tuple[np.ndarray, ResizeTransform]:
    """Resize ``image`` into a ``target``-sized canvas without distorting its aspect ratio.

    Args:
        image: Source image as an ``(H, W, 3)`` uint8 array.
        target: Canvas dimensions the detector expects.
        center: Place the scaled image at the canvas center instead of the top-left
            corner. SCRFD-family detectors are trained with top-left placement, so that
            is the default; centering is available for adapters that need it.

    Returns:
        The padded canvas and the transform that produced it.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"expected an (H, W, 3) image array, got shape {image.shape}"
        raise ValueError(msg)

    source = ImageSize(width=int(image.shape[1]), height=int(image.shape[0]))
    scale = min(target.width / source.width, target.height / source.height)
    scaled_width = max(1, round(source.width * scale))
    scaled_height = max(1, round(source.height * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)

    canvas = np.zeros((target.height, target.width, 3), dtype=image.dtype)
    pad_x = (target.width - scaled_width) // 2 if center else 0
    pad_y = (target.height - scaled_height) // 2 if center else 0
    canvas[pad_y : pad_y + scaled_height, pad_x : pad_x + scaled_width] = resized

    transform = ResizeTransform(
        source=source,
        canvas=target,
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
    )
    return canvas, transform


def stretch(image: np.ndarray, target: ImageSize) -> tuple[np.ndarray, ResizeTransform]:
    """Resize ``image`` directly to ``target`` dimensions without preserving aspect ratio.

    Args:
        image: Source image as an ``(H, W, 3)`` uint8 array.
        target: Canvas dimensions the model expects.

    Returns:
        The resized canvas and the transform that produced it.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"expected an (H, W, 3) image array, got shape {image.shape}"
        raise ValueError(msg)

    source = ImageSize(width=int(image.shape[1]), height=int(image.shape[0]))
    scale_x = target.width / source.width
    scale_y = target.height / source.height
    scale = min(scale_x, scale_y)

    interpolation = cv2.INTER_AREA if (scale_x < 1.0 or scale_y < 1.0) else cv2.INTER_LINEAR
    canvas = cv2.resize(image, (target.width, target.height), interpolation=interpolation)

    transform = ResizeTransform(
        source=source,
        canvas=target,
        scale=scale,
        pad_x=0.0,
        pad_y=0.0,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return canvas, transform
