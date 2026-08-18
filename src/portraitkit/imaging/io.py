"""Image loading at the input boundary.

Everything entering the pipeline passes through :func:`load_image`, which produces an
upright RGB array plus a record of what had to be corrected to get there. Centralizing
this means no stage has to wonder whether its input is BGR or RGB, sideways, palettized,
or partially truncated.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from portraitkit.errors import ImageLoadError
from portraitkit.imaging.orientation import OrientationFix, normalize_orientation
from portraitkit.types import ImageSize

__all__ = ["LoadedImage", "load_image"]

_TRUNCATION_MARKER = "truncated"


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """An upright RGB image and the provenance of the corrections applied to it."""

    pixels: np.ndarray
    """``(H, W, 3)`` uint8 array in RGB channel order, already upright."""

    orientation: OrientationFix
    """What EXIF orientation was found and whether pixels were transformed."""

    truncated: bool
    """Whether the file decoded only because truncated data was tolerated."""

    source: Path | None = None
    """Originating path, when the image was loaded from disk."""

    @property
    def size(self) -> ImageSize:
        """Dimensions of the upright image."""
        return ImageSize(width=int(self.pixels.shape[1]), height=int(self.pixels.shape[0]))

    def to_bgr(self) -> np.ndarray:
        """Return a BGR copy for the OpenCV-convention consumers that need one."""
        return self.pixels[:, :, ::-1].copy()


@contextmanager
def _tolerate_truncated_images(*, enabled: bool) -> Iterator[None]:
    """Temporarily set Pillow's global truncated-image policy.

    Pillow exposes this as process-global state, so this context manager is not
    thread-safe. Loading is expected to happen on one thread per process.
    """
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = enabled
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _decode(data: bytes | Path, *, tolerate_truncation: bool) -> Image.Image:
    handle = BytesIO(data) if isinstance(data, bytes) else data
    with _tolerate_truncated_images(enabled=tolerate_truncation):
        image = Image.open(handle)
        image.load()
    return image


def load_image(
    source: str | Path | bytes,
    *,
    tolerant: bool = True,
) -> LoadedImage:
    """Load ``source`` as an upright RGB image.

    The decode is attempted strictly first. Real-world enrollment archives contain JPEGs
    whose end-of-image marker is missing, which a strict decoder rejects outright even
    though every scanline is present; when ``tolerant`` is set, such a file is decoded on
    a second pass and flagged via :attr:`LoadedImage.truncated` so that callers and the
    evaluation harness can account for it instead of silently trusting it.

    Args:
        source: Filesystem path or raw encoded bytes.
        tolerant: Retry a truncated file with Pillow's permissive decoder.

    Returns:
        The loaded image, upright and in RGB order.

    Raises:
        ImageLoadError: If the file is missing, is not a recognizable image, or cannot
            be decoded even permissively.
    """
    path = None if isinstance(source, bytes) else Path(source)
    payload: bytes | Path = source if isinstance(source, bytes) else Path(source)
    truncated = False

    try:
        image = _decode(payload, tolerate_truncation=False)
    except OSError as error:
        if not tolerant or _TRUNCATION_MARKER not in str(error).lower():
            raise ImageLoadError(f"cannot decode image {path or '<bytes>'}: {error}") from error
        try:
            image = _decode(payload, tolerate_truncation=True)
        except (OSError, ValueError) as retry_error:
            raise ImageLoadError(
                f"cannot decode image {path or '<bytes>'} even permissively: {retry_error}"
            ) from retry_error
        truncated = True
    except (UnidentifiedImageError, ValueError) as error:
        raise ImageLoadError(f"cannot decode image {path or '<bytes>'}: {error}") from error

    upright, fix = normalize_orientation(image)
    rgb = upright if upright.mode == "RGB" else upright.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ImageLoadError(
            f"image {path or '<bytes>'} did not convert to three channels: shape {pixels.shape}"
        )

    return LoadedImage(pixels=pixels, orientation=fix, truncated=truncated, source=path)
