"""The four standard alpha-matting metrics.

Every expected value is hand-computable from the constructed matte, so a regression shows
up as a specific wrong number. The metrics are also checked against each other: each one
exists because it catches an error the others miss, and the tests assert those blind spots
directly rather than assuming them.
"""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.matting.metrics import (
    connectivity_error,
    gradient_error,
    matting_metrics,
    mean_squared_error,
    sum_absolute_difference,
)


def solid(value: float, size: int = 10) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float64)


def half_and_half(size: int = 10) -> np.ndarray:
    """A hard vertical edge: foreground on the left, background on the right."""
    matte = np.zeros((size, size), dtype=np.float64)
    matte[:, : size // 2] = 1.0
    return matte


# --- sum of absolute differences ------------------------------------------------------


def test_identical_mattes_have_zero_sad() -> None:
    matte = half_and_half()

    assert sum_absolute_difference(matte, matte) == pytest.approx(0.0)


def test_sad_matches_a_hand_computed_offset() -> None:
    """100 pixels each off by 0.1 is a raw sum of 10, reported in thousands as 0.01."""
    truth = solid(0.4)
    predicted = solid(0.5)

    assert sum_absolute_difference(predicted, truth) == pytest.approx(0.01)


def test_sad_is_symmetric() -> None:
    a, b = solid(0.25), solid(0.75)

    assert sum_absolute_difference(a, b) == pytest.approx(sum_absolute_difference(b, a))


def test_worst_case_sad_is_every_pixel_fully_wrong() -> None:
    assert sum_absolute_difference(solid(1.0), solid(0.0)) == pytest.approx(100 / 1000)


# --- mean squared error ---------------------------------------------------------------


def test_mse_matches_a_hand_computed_offset() -> None:
    """A uniform 0.1 error squares to 0.01 per pixel, and the mean of that is 0.01."""
    assert mean_squared_error(solid(0.5), solid(0.4)) == pytest.approx(0.01)


def test_mse_punishes_a_few_large_errors_more_than_many_small_ones() -> None:
    """This is why MSE is reported alongside SAD rather than instead of it."""
    truth = solid(0.0)
    diffuse = solid(0.1)
    concentrated = solid(0.0)
    concentrated[0, :] = 1.0

    # Both have the same raw absolute error of 10.
    assert sum_absolute_difference(diffuse, truth) == pytest.approx(
        sum_absolute_difference(concentrated, truth)
    )
    assert mean_squared_error(concentrated, truth) > mean_squared_error(diffuse, truth)


def test_mse_of_an_empty_region_is_zero_not_undefined() -> None:
    matte = solid(0.5)
    nothing = np.zeros_like(matte, dtype=bool)

    assert mean_squared_error(matte, matte, nothing) == 0.0


# --- gradient error -------------------------------------------------------------------


def test_identical_mattes_have_zero_gradient_error() -> None:
    matte = half_and_half()

    assert gradient_error(matte, matte) == pytest.approx(0.0)


def test_a_constant_offset_has_no_gradient_error() -> None:
    """The blind spot SAD has and gradient does not, in reverse.

    Shifting every alpha by a constant changes SAD but leaves every derivative untouched,
    so the gradient metric correctly reports no edge error.
    """
    truth = solid(0.4)
    predicted = solid(0.5)

    assert sum_absolute_difference(predicted, truth) > 0.0
    assert gradient_error(predicted, truth) == pytest.approx(0.0, abs=1e-9)


def test_a_smeared_edge_is_penalised_by_gradient() -> None:
    """The blind spot gradient exists to cover: a blurred edge can score well on SAD
    while looking obviously wrong, because the error is small but structural."""
    import cv2

    truth = half_and_half(32)
    smeared = cv2.GaussianBlur(truth, (0, 0), 3.0)

    assert gradient_error(smeared, truth) > 0.0


def test_gradient_error_is_symmetric() -> None:
    truth = half_and_half(16)
    other = np.roll(truth, 2, axis=1)

    assert gradient_error(other, truth) == pytest.approx(gradient_error(truth, other))


# --- connectivity error ---------------------------------------------------------------


def test_identical_mattes_have_zero_connectivity_error() -> None:
    matte = half_and_half(16)

    assert connectivity_error(matte, matte) == pytest.approx(0.0)


def test_a_detached_fragment_is_penalised() -> None:
    """The blind spot connectivity exists to cover: a small floating blob barely moves
    SAD but is the difference between a usable cut-out and an obviously broken one."""
    truth = np.zeros((32, 32), dtype=np.float64)
    truth[8:24, 8:24] = 1.0
    fragmented = truth.copy()
    fragmented[0:3, 0:3] = 1.0  # an island with no connection to the subject

    assert connectivity_error(fragmented, truth) > 0.0


def test_connectivity_ignores_a_uniform_matte_pair() -> None:
    matte = solid(1.0, 16)

    assert connectivity_error(matte, matte) == pytest.approx(0.0)


# --- trimap handling ------------------------------------------------------------------


def test_a_trimap_restricts_every_metric_to_its_unknown_region() -> None:
    """Scoring the known regions would reward a model for copying pixels it was given."""
    truth = solid(0.0, 10)
    predicted = solid(0.0, 10)
    predicted[0, :] = 1.0  # all the error sits in row 0

    unknown = np.zeros_like(truth, dtype=bool)
    unknown[5:, :] = True  # ... which the trimap excludes

    assert sum_absolute_difference(predicted, truth, unknown) == pytest.approx(0.0)
    assert mean_squared_error(predicted, truth, unknown) == pytest.approx(0.0)
    assert sum_absolute_difference(predicted, truth) > 0.0


def test_evaluated_pixel_count_reflects_the_trimap() -> None:
    truth = solid(0.5, 10)
    unknown = np.zeros_like(truth, dtype=bool)
    unknown[:4, :] = True

    assert matting_metrics(truth, truth, unknown).evaluated_pixels == 40


# --- combined -------------------------------------------------------------------------


def test_a_perfect_matte_scores_zero_on_everything() -> None:
    truth = half_and_half(16)

    metrics = matting_metrics(truth, truth)

    assert metrics.sad == pytest.approx(0.0)
    assert metrics.mse == pytest.approx(0.0)
    assert metrics.gradient == pytest.approx(0.0)
    assert metrics.connectivity == pytest.approx(0.0)
    assert metrics.evaluated_pixels == 256


def test_metrics_serialize() -> None:
    payload = matting_metrics(solid(0.5), solid(0.4)).to_dict()

    assert payload["sad"] == pytest.approx(0.01)
    assert payload["mse"] == pytest.approx(0.01)
    assert payload["evaluated_pixels"] == 100


# --- validation -----------------------------------------------------------------------


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        sum_absolute_difference(solid(0.5, 8), solid(0.5, 10))


def test_non_two_dimensional_mattes_are_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sum_absolute_difference(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))


@pytest.mark.parametrize("bad", [-0.5, 1.5])
def test_alpha_outside_the_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        sum_absolute_difference(solid(bad), solid(0.5))


def test_mismatched_trimap_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match the matte shape"):
        sum_absolute_difference(solid(0.5, 10), solid(0.5, 10), np.ones((4, 4), dtype=bool))
