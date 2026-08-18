"""EXIF orientation normalization and the coordinate mapping it implies.

Cameras and phones routinely record a sensor-native image plus an EXIF orientation tag
rather than rotating pixels. A detector fed the raw pixels sees a sideways or mirrored
face and either misses it or returns landmarks in the wrong frame. PortraitKit therefore
normalizes orientation once, at the input boundary, and every downstream stage works in
upright coordinates only.

Normalization is lossless and invertible. :func:`point_to_original` and
:func:`box_to_original` map upright coordinates back to the source pixel grid, so a
caller that needs to annotate the original file can still do so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from PIL import Image

from portraitkit.types import BoundingBox, ImageSize, Point

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = [
    "EXIF_ORIENTATION_TAG",
    "ExifOrientation",
    "OrientationFix",
    "box_to_original",
    "normalize_orientation",
    "point_to_original",
    "point_to_upright",
    "read_exif_orientation",
]

EXIF_ORIENTATION_TAG = 0x0112
"""The TIFF/EXIF tag number that carries the orientation value."""


class ExifOrientation(IntEnum):
    """The eight EXIF orientation values, named after the position of the 0th row/column.

    The names follow the TIFF specification: :attr:`RIGHT_TOP` means the 0th row of the
    stored image is on the right side of the displayed scene and the 0th column is at
    the top, which is what a camera writes when held rotated 90 degrees.
    """

    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4
    LEFT_TOP = 5
    RIGHT_TOP = 6
    RIGHT_BOTTOM = 7
    LEFT_BOTTOM = 8

    @property
    def is_identity(self) -> bool:
        """Whether the stored pixels are already upright."""
        return self is ExifOrientation.TOP_LEFT

    @property
    def swaps_axes(self) -> bool:
        """Whether normalizing exchanges image width and height."""
        return self.value >= ExifOrientation.LEFT_TOP.value

    @property
    def is_mirrored(self) -> bool:
        """Whether normalizing includes a reflection as well as a rotation."""
        return self in _MIRRORED


_MIRRORED = frozenset(
    {
        ExifOrientation.TOP_RIGHT,
        ExifOrientation.BOTTOM_LEFT,
        ExifOrientation.LEFT_TOP,
        ExifOrientation.RIGHT_BOTTOM,
    }
)

_TRANSPOSITIONS: dict[ExifOrientation, Image.Transpose] = {
    ExifOrientation.TOP_RIGHT: Image.Transpose.FLIP_LEFT_RIGHT,
    ExifOrientation.BOTTOM_RIGHT: Image.Transpose.ROTATE_180,
    ExifOrientation.BOTTOM_LEFT: Image.Transpose.FLIP_TOP_BOTTOM,
    ExifOrientation.LEFT_TOP: Image.Transpose.TRANSPOSE,
    ExifOrientation.RIGHT_TOP: Image.Transpose.ROTATE_270,
    ExifOrientation.RIGHT_BOTTOM: Image.Transpose.TRANSVERSE,
    ExifOrientation.LEFT_BOTTOM: Image.Transpose.ROTATE_90,
}


@dataclass(frozen=True, slots=True)
class OrientationFix:
    """A record of the orientation normalization applied to one image."""

    original: ExifOrientation
    source_size: ImageSize
    """Pixel dimensions as stored in the file, before normalization."""

    upright_size: ImageSize
    """Pixel dimensions after normalization."""

    @property
    def applied(self) -> bool:
        """Whether any pixel transform was actually performed."""
        return not self.original.is_identity


def read_exif_orientation(image: PILImage) -> ExifOrientation:
    """Return the EXIF orientation of ``image``.

    Images without EXIF data, without the orientation tag, or with a value outside the
    documented range 1-8 are reported as :attr:`ExifOrientation.TOP_LEFT`. A malformed
    tag is a reason to trust the pixels, not a reason to fail the load.
    """
    try:
        exif = image.getexif()
    except (AttributeError, OSError, ValueError):
        return ExifOrientation.TOP_LEFT
    raw = exif.get(EXIF_ORIENTATION_TAG)
    try:
        return ExifOrientation(int(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ExifOrientation.TOP_LEFT


def normalize_orientation(image: PILImage) -> tuple[PILImage, OrientationFix]:
    """Return ``image`` rotated and mirrored upright, with a record of what was done.

    The orientation tag is removed from the returned image so that a later consumer
    cannot apply the same correction twice.
    """
    orientation = read_exif_orientation(image)
    source_size = ImageSize(width=image.width, height=image.height)
    transposition = _TRANSPOSITIONS.get(orientation)
    if transposition is None:
        return image, OrientationFix(
            original=orientation, source_size=source_size, upright_size=source_size
        )

    upright = image.transpose(transposition)
    exif = upright.getexif()
    if EXIF_ORIENTATION_TAG in exif:
        del exif[EXIF_ORIENTATION_TAG]
        upright.info["exif"] = exif.tobytes()
    fix = OrientationFix(
        original=orientation,
        source_size=source_size,
        upright_size=ImageSize(width=upright.width, height=upright.height),
    )
    return upright, fix


def point_to_upright(point: Point, orientation: ExifOrientation, source_size: ImageSize) -> Point:
    """Map a point from stored-file coordinates into upright coordinates.

    Args:
        point: Position on the stored pixel grid.
        orientation: The EXIF orientation of the stored file.
        source_size: Dimensions of the stored file, before normalization.
    """
    x, y = point.x, point.y
    width, height = float(source_size.width), float(source_size.height)
    match orientation:
        case ExifOrientation.TOP_LEFT:
            return Point(x=x, y=y)
        case ExifOrientation.TOP_RIGHT:
            return Point(x=width - x, y=y)
        case ExifOrientation.BOTTOM_RIGHT:
            return Point(x=width - x, y=height - y)
        case ExifOrientation.BOTTOM_LEFT:
            return Point(x=x, y=height - y)
        case ExifOrientation.LEFT_TOP:
            return Point(x=y, y=x)
        case ExifOrientation.RIGHT_TOP:
            return Point(x=height - y, y=x)
        case ExifOrientation.RIGHT_BOTTOM:
            return Point(x=height - y, y=width - x)
        case ExifOrientation.LEFT_BOTTOM:
            return Point(x=y, y=width - x)


def point_to_original(point: Point, orientation: ExifOrientation, upright_size: ImageSize) -> Point:
    """Map a point from upright coordinates back to stored-file coordinates.

    Args:
        point: Position on the upright pixel grid.
        orientation: The EXIF orientation of the stored file.
        upright_size: Dimensions after normalization.
    """
    x, y = point.x, point.y
    width, height = float(upright_size.width), float(upright_size.height)
    match orientation:
        case ExifOrientation.TOP_LEFT:
            return Point(x=x, y=y)
        case ExifOrientation.TOP_RIGHT:
            return Point(x=width - x, y=y)
        case ExifOrientation.BOTTOM_RIGHT:
            return Point(x=width - x, y=height - y)
        case ExifOrientation.BOTTOM_LEFT:
            return Point(x=x, y=height - y)
        case ExifOrientation.LEFT_TOP:
            return Point(x=y, y=x)
        case ExifOrientation.RIGHT_TOP:
            return Point(x=y, y=width - x)
        case ExifOrientation.RIGHT_BOTTOM:
            return Point(x=height - y, y=width - x)
        case ExifOrientation.LEFT_BOTTOM:
            return Point(x=height - y, y=x)


def box_to_original(
    box: BoundingBox, orientation: ExifOrientation, upright_size: ImageSize
) -> BoundingBox:
    """Map an axis-aligned box from upright coordinates back to stored-file coordinates.

    Rotation and reflection can exchange which corner is the minimum, so the mapped
    corners are re-sorted rather than assumed.
    """
    corners = (
        point_to_original(Point(x=box.x1, y=box.y1), orientation, upright_size),
        point_to_original(Point(x=box.x2, y=box.y2), orientation, upright_size),
    )
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    return BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
