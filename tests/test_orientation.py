"""EXIF orientation normalization and its coordinate mapping.

The central test here checks the analytic mapping tables in
:mod:`portraitkit.imaging.orientation` against the pixels Pillow actually produces, for
all eight orientation values. An off-by-one or a swapped axis in those tables would put
every landmark in the wrong place for rotated phone photos, and would do so silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from portraitkit.imaging.orientation import (
    EXIF_ORIENTATION_TAG,
    ExifOrientation,
    box_to_original,
    normalize_orientation,
    point_to_original,
    point_to_upright,
    read_exif_orientation,
)
from portraitkit.types import BoundingBox, ImageSize, Point
from tests.conftest import CODED_HEIGHT, CODED_WIDTH, coded_image

# The transform that turns an upright image into what a camera would have stored for a
# given orientation tag: the inverse of the correction Pillow applies when displaying it.
INVERSE_TRANSPOSITION: dict[ExifOrientation, Image.Transpose | None] = {
    ExifOrientation.TOP_LEFT: None,
    ExifOrientation.TOP_RIGHT: Image.Transpose.FLIP_LEFT_RIGHT,
    ExifOrientation.BOTTOM_RIGHT: Image.Transpose.ROTATE_180,
    ExifOrientation.BOTTOM_LEFT: Image.Transpose.FLIP_TOP_BOTTOM,
    ExifOrientation.LEFT_TOP: Image.Transpose.TRANSPOSE,
    ExifOrientation.RIGHT_TOP: Image.Transpose.ROTATE_90,
    ExifOrientation.RIGHT_BOTTOM: Image.Transpose.TRANSVERSE,
    ExifOrientation.LEFT_BOTTOM: Image.Transpose.ROTATE_270,
}

ALL_ORIENTATIONS = list(ExifOrientation)


def _stored_for(orientation: ExifOrientation) -> Image.Image:
    """Synthesize the file a camera would have written for ``orientation``."""
    upright = coded_image()
    transposition = INVERSE_TRANSPOSITION[orientation]
    stored = upright if transposition is None else upright.transpose(transposition)
    stored.getexif()[EXIF_ORIENTATION_TAG] = int(orientation)
    return stored


def test_read_orientation_defaults_to_identity_without_exif() -> None:
    assert read_exif_orientation(coded_image()) is ExifOrientation.TOP_LEFT


@pytest.mark.parametrize("bogus", [0, 9, 255, -1, "six"])
def test_read_orientation_tolerates_malformed_tag(bogus: object) -> None:
    image = coded_image()
    image.getexif()[EXIF_ORIENTATION_TAG] = bogus
    assert read_exif_orientation(image) is ExifOrientation.TOP_LEFT


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_normalize_restores_the_upright_pixels(orientation: ExifOrientation) -> None:
    expected = np.asarray(coded_image())
    upright, fix = normalize_orientation(_stored_for(orientation))

    assert np.array_equal(np.asarray(upright), expected)
    assert fix.original is orientation
    assert fix.applied is (orientation is not ExifOrientation.TOP_LEFT)
    assert fix.upright_size == ImageSize(width=CODED_WIDTH, height=CODED_HEIGHT)


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_normalize_reports_swapped_dimensions(orientation: ExifOrientation) -> None:
    stored = _stored_for(orientation)
    _, fix = normalize_orientation(stored)

    if orientation.swaps_axes:
        assert fix.source_size == fix.upright_size.swapped()
    else:
        assert fix.source_size == fix.upright_size


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_normalize_clears_the_orientation_tag(orientation: ExifOrientation) -> None:
    upright, _ = normalize_orientation(_stored_for(orientation))
    assert read_exif_orientation(upright) is ExifOrientation.TOP_LEFT


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_point_to_upright_matches_the_actual_pixel_transform(
    orientation: ExifOrientation,
) -> None:
    """Every stored pixel must map to the upright pixel that holds its value."""
    stored = _stored_for(orientation)
    stored_pixels = np.asarray(stored)
    upright_pixels = np.asarray(coded_image())
    source_size = ImageSize(width=stored.width, height=stored.height)

    for stored_y in range(stored.height):
        for stored_x in range(stored.width):
            mapped = point_to_upright(
                Point(x=stored_x + 0.5, y=stored_y + 0.5), orientation, source_size
            )
            upright_x = math.floor(mapped.x)
            upright_y = math.floor(mapped.y)
            assert np.array_equal(
                upright_pixels[upright_y, upright_x], stored_pixels[stored_y, stored_x]
            ), f"{orientation.name}: stored ({stored_x}, {stored_y}) landed wrong"


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_point_mapping_round_trips(orientation: ExifOrientation) -> None:
    upright_size = ImageSize(width=CODED_WIDTH, height=CODED_HEIGHT)
    source_size = upright_size.swapped() if orientation.swaps_axes else upright_size

    for point in (Point(x=0.0, y=0.0), Point(x=1.5, y=2.5), Point(x=6.25, y=3.75)):
        original = point_to_original(point, orientation, upright_size)
        restored = point_to_upright(original, orientation, source_size)
        assert restored.x == pytest.approx(point.x)
        assert restored.y == pytest.approx(point.y)


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_box_to_original_stays_well_formed_and_preserves_area(
    orientation: ExifOrientation,
) -> None:
    upright_size = ImageSize(width=CODED_WIDTH, height=CODED_HEIGHT)
    box = BoundingBox(x1=1.0, y1=0.5, x2=5.0, y2=3.0)

    mapped = box_to_original(box, orientation, upright_size)

    assert mapped.x2 >= mapped.x1
    assert mapped.y2 >= mapped.y1
    assert mapped.area == pytest.approx(box.area)
    if orientation.swaps_axes:
        assert mapped.width == pytest.approx(box.height)
    else:
        assert mapped.width == pytest.approx(box.width)


def test_identity_orientation_is_a_no_op() -> None:
    image = coded_image()
    upright, fix = normalize_orientation(image)

    assert upright is image
    assert not fix.applied
    assert not fix.original.swaps_axes


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_mirrored_flag_matches_determinant_of_the_transform(
    orientation: ExifOrientation,
) -> None:
    """A reflection flips handedness; a pure rotation does not."""
    size = ImageSize(width=CODED_WIDTH, height=CODED_HEIGHT)
    source_size = size.swapped() if orientation.swaps_axes else size

    origin = point_to_upright(Point(x=0.0, y=0.0), orientation, source_size)
    along_x = point_to_upright(Point(x=1.0, y=0.0), orientation, source_size)
    along_y = point_to_upright(Point(x=0.0, y=1.0), orientation, source_size)
    determinant = (along_x.x - origin.x) * (along_y.y - origin.y) - (along_x.y - origin.y) * (
        along_y.x - origin.x
    )

    assert (determinant < 0) is orientation.is_mirrored
