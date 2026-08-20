"""Unit tests for the PortraitBench parameterized degradation suite."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.bench.degradations import (
    ClutteredBackground,
    Downscale,
    GaussianBlur,
    GaussianNoise,
    JpegCompression,
    LowLight,
    MotionBlur,
    apply_degradations,
    build_degradation,
)
from tests.conftest import solid_image


def test_jpeg_compression_degrades_image_and_preserves_shape() -> None:
    image = solid_image(width=64, height=64, color=(120, 180, 240))
    deg = JpegCompression(quality=20)

    out = deg.apply(image)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert deg.name == "jpeg_compression"
    assert deg.severity == "q20"


def test_jpeg_compression_rejects_invalid_quality() -> None:
    with pytest.raises(ValueError, match="quality must be in"):
        JpegCompression(quality=0)
    with pytest.raises(ValueError, match="quality must be in"):
        JpegCompression(quality=101)


def test_gaussian_blur_smooths_edges() -> None:
    # High frequency step edge
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :20] = 255
    deg = GaussianBlur(sigma=3.0)

    out = deg.apply(image)

    assert out.shape == image.shape
    # Center pixel should now be blurred towards middle value
    assert 50 < out[20, 20, 0] < 200


def test_motion_blur_applies_directional_filter() -> None:
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[15, 15] = 255
    deg = MotionBlur(kernel_size=9, angle_degrees=0.0)

    out = deg.apply(image)

    assert out.shape == image.shape
    # Horizontal line should be smeared across row 15
    assert np.any(out[15, 11:20, 0] > 0)


def test_low_light_reduces_luminance() -> None:
    image = np.full((30, 30, 3), 200, dtype=np.uint8)
    deg = LowLight(factor=0.5, gamma=1.0)

    out = deg.apply(image)

    assert out.shape == image.shape
    assert np.allclose(out, 100, atol=2)


def test_gaussian_noise_adds_deterministic_perturbation() -> None:
    image = np.full((30, 30, 3), 128, dtype=np.uint8)
    deg1 = GaussianNoise(std=20.0, seed=123)
    deg2 = GaussianNoise(std=20.0, seed=123)

    out1 = deg1.apply(image)
    out2 = deg2.apply(image)

    assert out1.shape == image.shape
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, image)


def test_downscale_reduces_and_restores_spatial_grid() -> None:
    image = solid_image(width=100, height=80)
    deg = Downscale(scale=0.25)

    out = deg.apply(image)

    assert out.shape == (80, 100, 3)
    assert out.dtype == np.uint8


def test_cluttered_background_injects_high_frequency_patterns() -> None:
    image = solid_image(width=100, height=100, color=(100, 100, 100))
    deg = ClutteredBackground(pattern="checkerboard", frequency=10)

    out = deg.apply(image)

    assert out.shape == (100, 100, 3)
    # Border area should be modified with clutter texture
    assert not np.array_equal(out[:10, :10], image[:10, :10])


def test_build_degradation_from_dict() -> None:
    deg = build_degradation({"type": "jpeg", "quality": 30})
    assert isinstance(deg, JpegCompression)
    assert deg.quality == 30

    deg = build_degradation({"type": "gaussian_blur", "sigma": 2.5})
    assert isinstance(deg, GaussianBlur)
    assert deg.sigma == pytest.approx(2.5)


def test_build_degradation_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown degradation type"):
        build_degradation({"type": "solar_flare"})


def test_apply_degradations_chains_multiple_steps() -> None:
    image = solid_image(width=40, height=40, color=(150, 150, 150))
    degs = [
        JpegCompression(quality=50),
        LowLight(factor=0.8),
    ]

    out = apply_degradations(image, degs)

    assert out.shape == image.shape
    assert np.mean(out) < np.mean(image)
