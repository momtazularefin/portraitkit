"""Core value objects."""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.types import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    Diagnostic,
    FaceDetection,
    FaceLandmarks5,
    ImageSize,
    Point,
)


def test_image_size_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ImageSize(width=0, height=10)


def test_image_size_swapped_and_aspect() -> None:
    size = ImageSize(width=200, height=100)

    assert size.swapped() == ImageSize(width=100, height=200)
    assert size.aspect_ratio == pytest.approx(2.0)


def test_bounding_box_geometry() -> None:
    box = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)

    assert box.width == pytest.approx(20.0)
    assert box.height == pytest.approx(40.0)
    assert box.area == pytest.approx(800.0)
    assert box.center == Point(x=20.0, y=40.0)


def test_bounding_box_rejects_inverted_corners() -> None:
    with pytest.raises(ValueError, match="x2 >= x1"):
        BoundingBox(x1=30.0, y1=0.0, x2=10.0, y2=10.0)


def test_bounding_box_clipping_to_image() -> None:
    size = ImageSize(width=100, height=100)
    box = BoundingBox(x1=-20.0, y1=-10.0, x2=120.0, y2=50.0)

    clipped = box.clipped_to(size)

    assert clipped.as_tuple() == (0.0, 0.0, 100.0, 50.0)
    assert not box.is_inside(size)
    assert clipped.is_inside(size)


def test_box_entirely_outside_collapses_to_zero_area() -> None:
    size = ImageSize(width=100, height=100)
    box = BoundingBox(x1=200.0, y1=200.0, x2=300.0, y2=300.0)

    clipped = box.clipped_to(size)

    assert clipped.area == pytest.approx(0.0)


def test_iou_of_identical_boxes_is_one() -> None:
    box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)

    assert box.intersection_over_union(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    left = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    right = BoundingBox(x1=20.0, y1=20.0, x2=30.0, y2=30.0)

    assert left.intersection_over_union(right) == pytest.approx(0.0)


def test_iou_of_half_overlap() -> None:
    left = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    right = BoundingBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0)

    # intersection 50, union 150
    assert left.intersection_over_union(right) == pytest.approx(1 / 3)


def test_iou_with_degenerate_boxes_is_zero() -> None:
    empty = BoundingBox(x1=5.0, y1=5.0, x2=5.0, y2=5.0)

    assert empty.intersection_over_union(empty) == pytest.approx(0.0)


def _level_landmarks() -> FaceLandmarks5:
    return FaceLandmarks5.from_array(
        np.asarray(
            [[40.0, 50.0], [60.0, 50.0], [50.0, 62.0], [42.0, 74.0], [58.0, 74.0]],
            dtype=np.float32,
        )
    )


def test_landmarks_from_array_preserves_detector_point_order() -> None:
    landmarks = _level_landmarks()

    assert landmarks.left_eye == Point(x=40.0, y=50.0)
    assert landmarks.right_eye == Point(x=60.0, y=50.0)
    assert landmarks.nose == Point(x=50.0, y=62.0)
    assert len(landmarks.as_points()) == 5


def test_landmarks_reject_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        FaceLandmarks5.from_array(np.zeros((3, 2), dtype=np.float32))


def test_level_eyes_give_zero_roll_and_correct_eye_line() -> None:
    landmarks = _level_landmarks()

    assert landmarks.roll_degrees == pytest.approx(0.0)
    assert landmarks.interocular_distance == pytest.approx(20.0)
    assert landmarks.eye_center == Point(x=50.0, y=50.0)


def test_roll_is_positive_when_the_right_eye_sits_lower() -> None:
    landmarks = FaceLandmarks5.from_array(
        np.asarray(
            [[40.0, 50.0], [60.0, 70.0], [50.0, 62.0], [42.0, 74.0], [58.0, 74.0]],
            dtype=np.float32,
        )
    )

    assert landmarks.roll_degrees == pytest.approx(45.0)


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_detection_rejects_out_of_range_scores(score: float) -> None:
    with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
        FaceDetection(box=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0), score=score)


def test_detection_result_reports_no_face_state() -> None:
    result = DetectionResult(
        status=DetectionStatus.NO_FACE, image_size=ImageSize(width=10, height=10)
    )

    assert not result.ok
    assert result.face_count == 0
    assert result.primary is None


def test_detection_result_reports_diagnostics() -> None:
    face = FaceDetection(box=BoundingBox(x1=0.0, y1=0.0, x2=5.0, y2=5.0), score=0.9)
    result = DetectionResult(
        status=DetectionStatus.OK,
        image_size=ImageSize(width=10, height=10),
        faces=(face, face),
        primary=face,
        diagnostics=(Diagnostic.MULTIPLE_FACES,),
    )

    assert result.ok
    assert result.face_count == 2
    assert result.has(Diagnostic.MULTIPLE_FACES)
    assert not result.has(Diagnostic.SMALL_FACE)


def test_diagnostic_values_are_stable_strings() -> None:
    """Diagnostics are serialized into benchmark output, so their values are an API."""
    assert Diagnostic.MULTIPLE_FACES == "multiple_faces"
    assert DetectionStatus.NO_FACE == "no_face"
