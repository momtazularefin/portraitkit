"""Image loading at the input boundary."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from portraitkit.errors import ImageLoadError
from portraitkit.imaging.io import load_image
from portraitkit.imaging.orientation import EXIF_ORIENTATION_TAG, ExifOrientation
from portraitkit.types import ImageSize
from tests.conftest import solid_image


def _write_jpeg(path: Path, pixels: np.ndarray, orientation: int | None = None) -> Path:
    image = Image.fromarray(pixels, mode="RGB")
    if orientation is None:
        image.save(path, format="JPEG", quality=95)
    else:
        exif = image.getexif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        image.save(path, format="JPEG", quality=95, exif=exif)
    return path


def test_load_from_path_returns_upright_rgb(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "plain.jpg", solid_image(width=32, height=16))

    loaded = load_image(path)

    assert loaded.pixels.dtype == np.uint8
    assert loaded.pixels.shape == (16, 32, 3)
    assert loaded.size == ImageSize(width=32, height=16)
    assert loaded.orientation.original is ExifOrientation.TOP_LEFT
    assert not loaded.orientation.applied
    assert not loaded.truncated
    assert loaded.source == path


def test_load_from_bytes_has_no_source(jpeg_bytes: bytes) -> None:
    loaded = load_image(jpeg_bytes)

    assert loaded.source is None
    assert loaded.size == ImageSize(width=64, height=48)


def test_load_applies_exif_rotation_and_swaps_dimensions(tmp_path: Path) -> None:
    # A 32x16 landscape frame stored with orientation 6 displays as 16x32 portrait.
    path = _write_jpeg(
        tmp_path / "rotated.jpg",
        solid_image(width=32, height=16),
        orientation=int(ExifOrientation.RIGHT_TOP),
    )

    loaded = load_image(path)

    assert loaded.size == ImageSize(width=16, height=32)
    assert loaded.orientation.original is ExifOrientation.RIGHT_TOP
    assert loaded.orientation.applied
    assert loaded.orientation.source_size == ImageSize(width=32, height=16)


def test_load_converts_grayscale_to_three_channels(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((12, 20), 90, dtype=np.uint8), mode="L").save(path)

    loaded = load_image(path)

    assert loaded.pixels.shape == (12, 20, 3)
    assert np.array_equal(loaded.pixels[:, :, 0], loaded.pixels[:, :, 2])


def test_load_drops_alpha_channel(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    Image.fromarray(np.full((8, 8, 4), 200, dtype=np.uint8), mode="RGBA").save(path)

    loaded = load_image(path)

    assert loaded.pixels.shape == (8, 8, 3)


def test_truncated_jpeg_is_recovered_and_flagged(tmp_path: Path, jpeg_bytes: bytes) -> None:
    """Enrollment archives contain JPEGs missing their end-of-image marker.

    Every scanline is present, so the picture is usable; a strict decoder still refuses
    it. PortraitKit recovers the image and records that it had to.
    """
    assert jpeg_bytes.endswith(b"\xff\xd9"), "fixture must carry a real EOI marker to remove"
    path = tmp_path / "truncated.jpg"
    path.write_bytes(jpeg_bytes[:-2])

    loaded = load_image(path)

    assert loaded.truncated
    assert loaded.pixels.shape == (48, 64, 3)


def test_truncated_jpeg_is_rejected_when_tolerance_is_off(
    tmp_path: Path, jpeg_bytes: bytes
) -> None:
    path = tmp_path / "truncated.jpg"
    path.write_bytes(jpeg_bytes[:-2])

    with pytest.raises(ImageLoadError, match="cannot decode"):
        load_image(path, tolerant=False)


def test_tolerant_decoding_does_not_leak_into_later_loads(
    tmp_path: Path, jpeg_bytes: bytes
) -> None:
    """The permissive flag is process-global in Pillow; it must be restored."""
    from PIL import ImageFile

    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(jpeg_bytes[:-2])
    load_image(truncated)

    assert ImageFile.LOAD_TRUNCATED_IMAGES is False

    with pytest.raises(ImageLoadError):
        load_image(truncated, tolerant=False)


def test_missing_file_raises_image_load_error(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="cannot decode"):
        load_image(tmp_path / "absent.jpg")


def test_non_image_payload_raises_image_load_error() -> None:
    with pytest.raises(ImageLoadError, match="cannot decode"):
        load_image(b"this is not an image")


def test_to_bgr_reverses_channel_order(jpeg_bytes: bytes) -> None:
    loaded = load_image(jpeg_bytes)

    bgr = loaded.to_bgr()

    assert np.array_equal(bgr[:, :, ::-1], loaded.pixels)


def test_empty_payload_raises_image_load_error() -> None:
    buffer = BytesIO()
    with pytest.raises(ImageLoadError):
        load_image(buffer.getvalue())
