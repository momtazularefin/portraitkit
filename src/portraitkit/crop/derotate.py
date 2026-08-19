"""Levelling the eye line before cropping.

A tilted head defeats an axis-aligned crop: the head occupies more vertical extent than
its true crown-to-chin length, so a rectangle sized from that length frames the subject
wrongly. Rotating about the eye centre until the eye line is level fixes it before any
geometry is solved.

Rotation is the right correction rather than a convenience. ICAO Doc 9303 Part 3, 3.9.1.2
requires modification by cropping and forbids stretching, and a rigid rotation preserves
every ratio the standard uses to detect stretching, including inter-eye to eye-to-mouth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from portraitkit.types import FaceLandmarks5

__all__ = ["Derotation", "level_eye_line"]


@dataclass(frozen=True, slots=True)
class Derotation:
    """A record of the levelling applied to one image."""

    applied: bool
    angle_degrees: float
    """Rotation used, in the convention of :func:`cv2.getRotationMatrix2D`. Zero when the
    tilt was already inside tolerance."""

    original_roll_degrees: float


def _transform_landmarks(landmarks: FaceLandmarks5, matrix: np.ndarray) -> FaceLandmarks5:
    points = np.asarray([[point.x, point.y] for point in landmarks.as_points()], dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return FaceLandmarks5.from_array((homogeneous @ matrix.T).astype(np.float32))


def level_eye_line(
    pixels: np.ndarray,
    landmarks: FaceLandmarks5,
    *,
    tolerance_degrees: float = 1.0,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, FaceLandmarks5, Derotation]:
    """Rotate ``pixels`` about the eye centre so the eye line becomes horizontal.

    Args:
        pixels: ``(H, W, 3)`` uint8 RGB image.
        landmarks: Landmarks in ``pixels`` coordinates.
        tolerance_degrees: Tilt at or below which no rotation is performed. Rotating for
            a fraction of a degree costs a resampling pass and buys nothing.
        background: Fill for the corners rotation leaves empty.

    Returns:
        The levelled image, the landmarks mapped into it, and a record of what was done.
        The image keeps its original dimensions; the crop stage handles any framing that
        subsequently falls outside it.
    """
    roll = landmarks.roll_degrees
    if abs(roll) <= tolerance_degrees:
        return (
            pixels,
            landmarks,
            Derotation(applied=False, angle_degrees=0.0, original_roll_degrees=roll),
        )

    centre = landmarks.eye_center
    # getRotationMatrix2D maps a vector (dx, dy) to (a*dx + b*dy, -b*dx + a*dy) with
    # a = cos(angle) and b = sin(angle). Solving -b*dx + a*dy = 0 for the eye vector
    # gives tan(angle) = dy/dx, so the angle that levels the eye line is the roll itself.
    matrix = cv2.getRotationMatrix2D((centre.x, centre.y), roll, 1.0)
    height, width = pixels.shape[:2]
    rotated = cv2.warpAffine(
        pixels,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(channel) for channel in background),
    )
    return (
        rotated,
        _transform_landmarks(landmarks, matrix),
        Derotation(applied=True, angle_degrees=roll, original_roll_degrees=roll),
    )
