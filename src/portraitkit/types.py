"""Typed value objects shared by every PortraitKit stage.

Coordinate convention
---------------------
All boxes and landmarks are expressed in pixel coordinates of the *upright* image:
the image after EXIF orientation has been normalized (see
:mod:`portraitkit.imaging.orientation`). The origin is the top-left corner, ``x`` grows
to the right and ``y`` grows downward.

``left`` and ``right`` in landmark names refer to the **image frame**, not the subject
anatomy: ``left_eye`` is the eye nearer the left edge of the picture, which for a
front-facing subject is anatomically their right eye. This matches the point order
emitted by the SCRFD/RetinaFace family of detectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "DetectionStatus",
    "Diagnostic",
    "FaceDetection",
    "FaceLandmarks5",
    "ImageSize",
    "Point",
]


@dataclass(frozen=True, slots=True)
class Point:
    """A point in upright-image pixel coordinates."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to ``other`` in pixels."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[float, float]:
        """Return the point as a plain ``(x, y)`` tuple."""
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class ImageSize:
    """Pixel dimensions of an image."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = f"image size must be positive, got {self.width}x{self.height}"
            raise ValueError(msg)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / self.height

    def swapped(self) -> ImageSize:
        """Return the size with width and height exchanged."""
        return ImageSize(width=self.height, height=self.width)

    def as_tuple(self) -> tuple[int, int]:
        """Return the size as a plain ``(width, height)`` tuple."""
        return (self.width, self.height)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in upright-image pixel coordinates.

    Coordinates are half-open: ``x1``/``y1`` are inclusive and ``x2``/``y2`` are
    exclusive, so ``width`` is simply ``x2 - x1``.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            msg = (
                "bounding box requires x2 >= x1 and y2 >= y1, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Geometric center of the box."""
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    def clipped_to(self, size: ImageSize) -> BoundingBox:
        """Return the box intersected with the image rectangle of ``size``.

        A box entirely outside the image collapses to a zero-area box on the nearest
        edge rather than raising, so callers can test ``area == 0``.
        """
        x1 = min(max(self.x1, 0.0), float(size.width))
        y1 = min(max(self.y1, 0.0), float(size.height))
        x2 = min(max(self.x2, 0.0), float(size.width))
        y2 = min(max(self.y2, 0.0), float(size.height))
        return BoundingBox(x1=x1, y1=y1, x2=max(x2, x1), y2=max(y2, y1))

    def is_inside(self, size: ImageSize) -> bool:
        """Whether the box lies entirely within the image rectangle of ``size``."""
        return (
            self.x1 >= 0.0
            and self.y1 >= 0.0
            and self.x2 <= float(size.width)
            and self.y2 <= float(size.height)
        )

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Intersection-over-union overlap with ``other``, in ``[0, 1]``."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the box as a plain ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class FaceLandmarks5:
    """The five-point landmark set emitted by SCRFD/RetinaFace-class detectors.

    See the module docstring for the ``left``/``right`` naming convention.
    """

    left_eye: Point
    right_eye: Point
    nose: Point
    left_mouth: Point
    right_mouth: Point

    @classmethod
    def from_array(cls, points: np.ndarray) -> FaceLandmarks5:
        """Build a landmark set from a ``(5, 2)`` array in detector point order."""
        if points.shape != (5, 2):
            msg = f"expected a (5, 2) landmark array, got {points.shape}"
            raise ValueError(msg)
        left_eye, right_eye, nose, left_mouth, right_mouth = (
            Point(x=float(x), y=float(y)) for x, y in points
        )
        return cls(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            left_mouth=left_mouth,
            right_mouth=right_mouth,
        )

    def as_points(self) -> tuple[Point, Point, Point, Point, Point]:
        """Return the landmarks in detector point order."""
        return (self.left_eye, self.right_eye, self.nose, self.left_mouth, self.right_mouth)

    @property
    def eye_center(self) -> Point:
        """Midpoint of the two eye landmarks, the ICAO eye-line reference."""
        return Point(
            x=(self.left_eye.x + self.right_eye.x) / 2,
            y=(self.left_eye.y + self.right_eye.y) / 2,
        )

    @property
    def interocular_distance(self) -> float:
        """Distance between the eye landmarks, the standard landmark-error normalizer."""
        return self.left_eye.distance_to(self.right_eye)

    @property
    def roll_degrees(self) -> float:
        """In-plane rotation of the eye line, positive clockwise in image space.

        Zero means the eyes are level. This is the roll signal the crop stage uses to
        decide whether an image needs de-rotation before an ICAO crop.
        """
        return math.degrees(
            math.atan2(self.right_eye.y - self.left_eye.y, self.right_eye.x - self.left_eye.x)
        )


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One detected face: its box, confidence, and optional landmarks."""

    box: BoundingBox
    score: float
    landmarks: FaceLandmarks5 | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            msg = f"detection score must be in [0, 1], got {self.score}"
            raise ValueError(msg)


class DetectionStatus(StrEnum):
    """Outcome of the detection stage.

    ``OK`` means at least one face passed the confidence threshold and a primary
    subject was selected. Anything else leaves :attr:`DetectionResult.primary` unset.
    """

    OK = "ok"
    NO_FACE = "no_face"


class Diagnostic(StrEnum):
    """Non-fatal observations attached to a stage result.

    These carry forward the failure vocabulary of the legacy capture pipeline so that
    downstream stages and the evaluation harness can report *why* an image was hard,
    not merely that it failed.
    """

    MULTIPLE_FACES = "multiple_faces"
    """More than one face passed the score threshold; a primary was selected."""

    LOW_CONFIDENCE = "low_confidence"
    """The primary face scored below the comfortable-confidence margin."""

    FACE_TOUCHES_BORDER = "face_touches_border"
    """The primary face box reaches an image edge and may be cut off."""

    SMALL_FACE = "small_face"
    """The primary face is small relative to the frame; the subject is likely distant."""

    ORIENTATION_CORRECTED = "orientation_corrected"
    """A non-identity EXIF orientation was applied before detection."""

    TRUNCATED_IMAGE_DATA = "truncated_image_data"
    """The source file decoded only under a tolerant decoder."""

    STRONG_ROLL = "strong_roll"
    """The eye line is far from level; the head is noticeably tilted in plane."""


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Typed result of the detection and orientation stage.

    The stage always returns one of these. An image with no usable face yields
    ``status=NO_FACE`` with an empty :attr:`faces` tuple rather than an exception.
    """

    status: DetectionStatus
    image_size: ImageSize
    faces: tuple[FaceDetection, ...] = ()
    primary: FaceDetection | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    detector: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether a primary face was selected."""
        return self.status is DetectionStatus.OK and self.primary is not None

    @property
    def face_count(self) -> int:
        """Number of faces that passed the score threshold."""
        return len(self.faces)

    def has(self, diagnostic: Diagnostic) -> bool:
        """Whether ``diagnostic`` was recorded on this result."""
        return diagnostic in self.diagnostics
