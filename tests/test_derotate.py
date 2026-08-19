"""Levelling the eye line before cropping.

The rotation sign is derived analytically in the implementation, so the tests check it
empirically against OpenCV rather than restating the derivation. A sign error here would
tilt every portrait the wrong way by twice the roll, which still produces a plausible
looking picture.
"""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.crop.derotate import level_eye_line
from portraitkit.crop.stage import CropConfig, CropStage
from portraitkit.types import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    FaceDetection,
    FaceLandmarks5,
    ImageSize,
)
from tests.conftest import solid_image


def tilted_landmarks(roll_degrees: float, centre: tuple[float, float] = (200.0, 200.0)):
    """Build a face whose eye line is tilted by ``roll_degrees``."""
    half = 20.0
    radians = np.radians(roll_degrees)
    dx, dy = half * np.cos(radians), half * np.sin(radians)
    cx, cy = centre
    return FaceLandmarks5.from_array(
        np.asarray(
            [
                [cx - dx, cy - dy],
                [cx + dx, cy + dy],
                [cx, cy + 25.0],
                [cx - 18.0, cy + 40.0],
                [cx + 18.0, cy + 40.0],
            ],
            dtype=np.float32,
        )
    )


def test_a_level_face_is_left_alone() -> None:
    pixels = solid_image(400, 400)
    landmarks = tilted_landmarks(0.0)

    result, moved, record = level_eye_line(pixels, landmarks, tolerance_degrees=1.0)

    assert not record.applied
    assert record.angle_degrees == 0.0
    assert result is pixels
    assert moved is landmarks


def test_tilt_inside_tolerance_is_left_alone() -> None:
    """Resampling for a fraction of a degree costs quality and buys nothing."""
    _, _, record = level_eye_line(
        solid_image(400, 400), tilted_landmarks(0.4), tolerance_degrees=1.0
    )

    assert not record.applied


@pytest.mark.parametrize("roll", [-30.0, -12.5, -3.0, 3.0, 12.5, 30.0])
def test_levelling_drives_roll_to_zero(roll: float) -> None:
    """The sign must be right in both directions, not merely consistent."""
    _, moved, record = level_eye_line(
        solid_image(400, 400), tilted_landmarks(roll), tolerance_degrees=1.0
    )

    assert record.applied
    assert moved.roll_degrees == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("roll", [-20.0, 20.0])
def test_levelling_preserves_the_eye_centre(roll: float) -> None:
    """Rotation is about the eye centre, so that point must not move."""
    landmarks = tilted_landmarks(roll)
    before = landmarks.eye_center

    _, moved, _ = level_eye_line(solid_image(400, 400), landmarks, tolerance_degrees=1.0)

    assert moved.eye_center.x == pytest.approx(before.x, abs=1e-3)
    assert moved.eye_center.y == pytest.approx(before.y, abs=1e-3)


@pytest.mark.parametrize("roll", [-25.0, 25.0])
def test_levelling_is_rigid(roll: float) -> None:
    """A rigid rotation preserves every distance, which is what keeps the crop honest
    against the no-stretch requirement in ICAO Doc 9303 Part 3, 3.9.1.2."""
    landmarks = tilted_landmarks(roll)

    _, moved, _ = level_eye_line(solid_image(400, 400), landmarks, tolerance_degrees=1.0)

    assert moved.interocular_distance == pytest.approx(landmarks.interocular_distance, rel=1e-4)
    before = landmarks.eye_center.distance_to(landmarks.nose)
    after = moved.eye_center.distance_to(moved.nose)
    assert after == pytest.approx(before, rel=1e-4)


def test_levelling_keeps_the_image_size() -> None:
    pixels = solid_image(400, 300)

    rotated, _, _ = level_eye_line(pixels, tilted_landmarks(20.0), tolerance_degrees=1.0)

    assert rotated.shape == pixels.shape


def test_corners_are_filled_with_the_background() -> None:
    rotated, _, _ = level_eye_line(
        solid_image(400, 400, color=(10, 10, 10)),
        tilted_landmarks(30.0),
        tolerance_degrees=1.0,
        background=(250, 0, 0),
    )

    assert rotated[0, 0, 0] > 200


# --- through the stage ----------------------------------------------------------------


def detection_with(landmarks: FaceLandmarks5, size: ImageSize) -> DetectionResult:
    face = FaceDetection(
        box=BoundingBox(x1=150.0, y1=150.0, x2=250.0, y2=270.0), score=0.95, landmarks=landmarks
    )
    return DetectionResult(status=DetectionStatus.OK, image_size=size, faces=(face,), primary=face)


def test_stage_levels_a_tilted_head_before_solving() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_with(tilted_landmarks(20.0), size))

    assert result.ok
    assert result.derotation is not None
    assert result.derotation.applied
    assert result.derotation.original_roll_degrees == pytest.approx(20.0, abs=0.1)
    assert result.metadata["derotated_degrees"] == pytest.approx(20.0, abs=0.1)


def test_levelling_can_be_disabled() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage(CropConfig(derotate=False)).run(
        solid_image(400, 400), detection_with(tilted_landmarks(20.0), size)
    )

    assert result.ok
    assert result.derotation is None


def test_a_tilted_head_frames_better_after_levelling() -> None:
    """A tilted head spans more vertical extent than its true crown-to-chin length, so
    leaving it tilted inflates the head length the solver works from."""
    size = ImageSize(width=400, height=400)
    tilted = detection_with(tilted_landmarks(25.0), size)

    levelled = CropStage().run(solid_image(400, 400), tilted)
    raw = CropStage(CropConfig(derotate=False)).run(solid_image(400, 400), tilted)

    assert levelled.plan is not None
    assert raw.plan is not None
    # A 25 degree tilt spreads the eye-to-mouth span across both axes, so the untilted
    # measurement is the smaller and truer one.
    assert levelled.plan.head.eye_to_mouth < raw.plan.head.eye_to_mouth
    assert levelled.plan.achieved_face_centre_vertical == pytest.approx(0.40)
