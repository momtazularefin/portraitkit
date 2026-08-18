"""Choosing the primary subject when a photo contains more than one face.

The legacy capture pipeline hard-coded largest-face selection. That is a reasonable
default for enrollment photography and a poor one for a group snapshot where the subject
stands slightly behind someone else, and the old code offered no way to tell which case
you were in. Here the rule is named, swappable, and reported on the stage result, so the
evaluation twin can measure it rather than inherit it.
"""

from __future__ import annotations

from enum import StrEnum

from portraitkit.types import FaceDetection, ImageSize

__all__ = ["SelectionStrategy", "select_primary"]


class SelectionStrategy(StrEnum):
    """How a primary subject is picked from several candidates."""

    LARGEST = "largest"
    """Largest box area. The conventional baseline, and the default until the M1
    evaluation says otherwise."""

    MOST_CENTRAL = "most_central"
    """Box centre nearest the frame centre."""

    MOST_CONFIDENT = "most_confident"
    """Highest detector confidence."""

    BALANCED = "balanced"
    """Weighted blend of relative size and centrality, for photos where the intended
    subject is prominent but not necessarily the largest face present."""


def _centrality(face: FaceDetection, size: ImageSize) -> float:
    """Return 1.0 at the frame centre, falling toward 0.0 at the corners."""
    centre = face.box.center
    offset_x = (centre.x - size.width / 2.0) / (size.width / 2.0)
    offset_y = (centre.y - size.height / 2.0) / (size.height / 2.0)
    distance = (offset_x**2 + offset_y**2) ** 0.5
    return max(0.0, 1.0 - distance / (2**0.5))


def select_primary(
    faces: tuple[FaceDetection, ...],
    image_size: ImageSize,
    strategy: SelectionStrategy = SelectionStrategy.LARGEST,
) -> FaceDetection | None:
    """Pick the primary subject, or ``None`` when there are no candidates.

    Ties are broken by detector confidence so the result is deterministic for a given
    input, which reproducible benchmark runs depend on.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]

    match strategy:
        case SelectionStrategy.LARGEST:
            key = lambda face: (face.box.area, face.score)  # noqa: E731
        case SelectionStrategy.MOST_CENTRAL:
            key = lambda face: (_centrality(face, image_size), face.score)  # noqa: E731
        case SelectionStrategy.MOST_CONFIDENT:
            key = lambda face: (face.score, face.box.area)  # noqa: E731
        case SelectionStrategy.BALANCED:
            frame_area = float(image_size.width * image_size.height)

            def key(face: FaceDetection) -> tuple[float, float]:
                relative_area = min(1.0, face.box.area / frame_area) ** 0.5
                blended = 0.6 * relative_area + 0.4 * _centrality(face, image_size)
                return (blended, face.score)

    return max(faces, key=key)
