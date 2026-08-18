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

__all__ = ["ResizeTransform", "letterbox"]


@dataclass(frozen=True, slots=True)
class ResizeTransform:
    """An aspect-preserving resize followed by a translation into a padded canvas.

    Forward direction maps source-image coordinates to canvas coordinates:
    ``canvas = source * scale + pad``. :meth:`invert_point` and :meth:`invert_box` undo
    it, which is what turns raw detector output back into image coordinates.
    """

    source: ImageSize
    canvas: ImageSize
    scale: float
    pad_x: float
    pad_y: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            msg = f"resize scale must be positive, got {self.scale}"
            raise ValueError(msg)

    def apply_point(self, point: Point) -> Point:
        """Map a source-image point onto the padded canvas."""
        return Point(x=point.x * self.scale + self.pad_x, y=point.y * self.scale + self.pad_y)

    def invert_point(self, point: Point) -> Point:
        """Map a canvas point back to source-image coordinates."""
        return Point(x=(point.x - self.pad_x) / self.scale, y=(point.y - self.pad_y) / self.scale)

    def invert_box(self, box: BoundingBox) -> BoundingBox:
        """Map a canvas box back to source-image coordinates."""
        top_left = self.invert_point(Point(x=box.x1, y=box.y1))
        bottom_right = self.invert_point(Point(x=box.x2, y=box.y2))
        return BoundingBox(x1=top_left.x, y1=top_left.y, x2=bottom_right.x, y2=bottom_right.y)

    def invert_array(self, points: np.ndarray) -> np.ndarray:
        """Map an ``(..., 2)`` array of canvas coordinates back to source coordinates."""
        offset = np.asarray([self.pad_x, self.pad_y], dtype=np.float32)
        return (np.asarray(points, dtype=np.float32) - offset) / self.scale


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
