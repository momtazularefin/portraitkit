"""Invertible resize transforms."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.imaging.geometry import ResizeTransform, letterbox
from portraitkit.types import BoundingBox, ImageSize, Point
from tests.conftest import solid_image


def test_letterbox_preserves_aspect_ratio_of_a_wide_image() -> None:
    image = solid_image(width=200, height=100)

    canvas, transform = letterbox(image, ImageSize(width=64, height=64))

    assert canvas.shape == (64, 64, 3)
    assert transform.scale == pytest.approx(0.32)
    assert transform.pad_x == 0.0
    assert transform.pad_y == 0.0
    # Content occupies the top 32 rows; the remainder is padding.
    assert canvas[:32].any()
    assert not canvas[32:].any()


def test_letterbox_centers_when_asked() -> None:
    image = solid_image(width=200, height=100)

    canvas, transform = letterbox(image, ImageSize(width=64, height=64), center=True)

    assert transform.pad_y == 16.0
    assert transform.pad_x == 0.0
    assert not canvas[:16].any()
    assert canvas[16:48].any()
    assert not canvas[48:].any()


def test_letterbox_upscales_a_small_image() -> None:
    image = solid_image(width=10, height=10)

    canvas, transform = letterbox(image, ImageSize(width=40, height=40))

    assert transform.scale == pytest.approx(4.0)
    assert canvas.shape == (40, 40, 3)


def test_letterbox_rejects_non_three_channel_input() -> None:
    with pytest.raises(ValueError, match="H, W, 3"):
        letterbox(np.zeros((8, 8), dtype=np.uint8), ImageSize(width=16, height=16))


def test_transform_round_trips_points() -> None:
    transform = ResizeTransform(
        source=ImageSize(width=200, height=100),
        canvas=ImageSize(width=64, height=64),
        scale=0.32,
        pad_x=0.0,
        pad_y=16.0,
    )
    point = Point(x=57.0, y=23.0)

    restored = transform.invert_point(transform.apply_point(point))

    assert restored.x == pytest.approx(point.x)
    assert restored.y == pytest.approx(point.y)


def test_transform_inverts_a_box_into_source_space() -> None:
    _, transform = letterbox(solid_image(width=200, height=100), ImageSize(width=64, height=64))
    canvas_box = BoundingBox(x1=16.0, y1=8.0, x2=32.0, y2=24.0)

    source_box = transform.invert_box(canvas_box)

    assert source_box.x1 == pytest.approx(50.0)
    assert source_box.y1 == pytest.approx(25.0)
    assert source_box.width == pytest.approx(50.0)
    assert source_box.height == pytest.approx(50.0)


def test_invert_array_matches_invert_point() -> None:
    transform = ResizeTransform(
        source=ImageSize(width=200, height=100),
        canvas=ImageSize(width=64, height=64),
        scale=0.32,
        pad_x=4.0,
        pad_y=16.0,
    )
    points = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    inverted = transform.invert_array(points)

    for row, (x, y) in zip(inverted, points, strict=True):
        expected = transform.invert_point(Point(x=float(x), y=float(y)))
        assert float(row[0]) == pytest.approx(expected.x, rel=1e-5)
        assert float(row[1]) == pytest.approx(expected.y, rel=1e-5)


def test_transform_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        ResizeTransform(
            source=ImageSize(width=4, height=4),
            canvas=ImageSize(width=4, height=4),
            scale=0.0,
            pad_x=0.0,
            pad_y=0.0,
        )
