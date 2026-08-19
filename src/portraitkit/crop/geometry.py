"""Head extent estimation and crop solving.

The requirement is stated in terms of crown and chin. A five-point landmark set contains
neither: it gives the eyes, the nose tip, and the mouth corners. Crown and chin are
therefore *estimated*, and this module keeps that fact explicit rather than burying it,
because a compliance claim resting on an estimate is not a compliance claim.

The estimate is anchored on two distances Doc 9303 itself names. Section 3.9.1.3 says
that where the printed image's height cannot be determined directly, the ratio between
the inter-eye distance (IED) and the eye-to-mouth distance (EM) is what must be
preserved. Both are directly measurable from the landmarks we have, which makes them the
natural basis for inferring the rest of the head.

The inference uses the classical proportion that the eye line sits near the vertical
midpoint of the head, and that the mouth sits about two thirds of the way from the eye
line to the chin. Those give eye-to-chin of roughly 1.5 EM and crown-to-eye of about the
same, so crown-to-chin is roughly 3 EM. These are population averages; individual heads
vary, which is precisely why the external referee in M2b exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from portraitkit.crop.presets import CropPreset
from portraitkit.types import BoundingBox, FaceLandmarks5, ImageSize, Point

__all__ = [
    "CROWN_TO_EYE_PER_EM",
    "EYE_TO_CHIN_PER_EM",
    "CropPlan",
    "HeadEstimate",
    "estimate_head",
    "solve_crop",
]

EYE_TO_CHIN_PER_EM: float = 1.5
"""Eye line to chin, in units of eye-to-mouth distance. Follows from the mouth sitting
about two thirds of the way from the eye line to the chin."""

CROWN_TO_EYE_PER_EM: float = 1.5
"""Eye line to crown, in units of eye-to-mouth distance. Follows from the eye line
sitting near the vertical midpoint of the head."""


@dataclass(frozen=True, slots=True)
class HeadEstimate:
    """Head extent inferred from landmarks.

    ``interocular_distance`` and ``eye_to_mouth`` are measured. ``crown``, ``chin`` and
    ``height`` are inferred from them and carry the uncertainty described in the module
    docstring.
    """

    eye_centre: Point
    mouth_centre: Point
    crown: Point
    chin: Point
    interocular_distance: float
    eye_to_mouth: float
    roll_degrees: float

    @property
    def height(self) -> float:
        """Estimated crown-to-chin extent in pixels."""
        return self.chin.y - self.crown.y

    @property
    def ied_to_em_ratio(self) -> float:
        """The ratio Doc 9303 3.9.1.3 requires a resize to preserve.

        Cropping and uniform scaling both leave it unchanged, so a departure from the
        source value indicates the image was stretched.
        """
        if self.eye_to_mouth <= 0.0:
            return 0.0
        return self.interocular_distance / self.eye_to_mouth


@dataclass(frozen=True, slots=True)
class CropPlan:
    """A crop rectangle in source-image coordinates, plus what it will take to realize it."""

    rect: BoundingBox
    """Region to extract. May extend beyond the source image; see :attr:`needs_padding`."""

    preset: CropPreset
    source_size: ImageSize
    head: HeadEstimate

    @property
    def scale(self) -> float:
        """Factor mapping the crop rectangle onto the preset's output size."""
        return self.preset.output_size.height / self.rect.height

    @property
    def needs_padding(self) -> bool:
        """Whether the rectangle reaches outside the source image.

        A correctly framed portrait often requires canvas the original photograph does
        not contain, particularly above the crown. Reporting it lets the caller decide
        between padding the background and rejecting the photo, instead of silently
        producing a crop that violates the geometry it claims to satisfy.
        """
        return not self.rect.is_inside(self.source_size)

    @property
    def padding(self) -> tuple[float, float, float, float]:
        """Pixels of canvas needed beyond each edge, as ``(left, top, right, bottom)``."""
        return (
            max(0.0, -self.rect.x1),
            max(0.0, -self.rect.y1),
            max(0.0, self.rect.x2 - self.source_size.width),
            max(0.0, self.rect.y2 - self.source_size.height),
        )

    @property
    def achieved_head_height_ratio(self) -> float:
        """Estimated crown-to-chin extent as a fraction of the crop height."""
        return self.head.height / self.rect.height


def estimate_head(landmarks: FaceLandmarks5) -> HeadEstimate:
    """Infer crown and chin from a five-point landmark set.

    Raises:
        ValueError: If the landmarks are degenerate, leaving no scale to work from.
    """
    eye_centre = landmarks.eye_center
    mouth_centre = Point(
        x=(landmarks.left_mouth.x + landmarks.right_mouth.x) / 2,
        y=(landmarks.left_mouth.y + landmarks.right_mouth.y) / 2,
    )
    eye_to_mouth = eye_centre.distance_to(mouth_centre)
    if eye_to_mouth <= 0.0:
        msg = "cannot estimate head extent: eye and mouth landmarks coincide"
        raise ValueError(msg)

    interocular = landmarks.interocular_distance
    if interocular <= 0.0:
        msg = "cannot estimate head extent: eye landmarks coincide"
        raise ValueError(msg)

    # Extend along the image vertical rather than along the head axis. The crop stage
    # de-rotates a strongly rolled head before cropping, so by this point the two are
    # close, and staying axis-aligned keeps the rectangle axis-aligned too.
    return HeadEstimate(
        eye_centre=eye_centre,
        mouth_centre=mouth_centre,
        crown=Point(x=eye_centre.x, y=eye_centre.y - CROWN_TO_EYE_PER_EM * eye_to_mouth),
        chin=Point(x=eye_centre.x, y=eye_centre.y + EYE_TO_CHIN_PER_EM * eye_to_mouth),
        interocular_distance=interocular,
        eye_to_mouth=eye_to_mouth,
        roll_degrees=landmarks.roll_degrees,
    )


def solve_crop(
    head: HeadEstimate,
    preset: CropPreset,
    source_size: ImageSize,
) -> CropPlan:
    """Compute the crop rectangle placing ``head`` as ``preset`` requires.

    The rectangle is sized so the estimated crown-to-chin extent occupies the preset's
    target fraction of the output height, then positioned to centre the head
    horizontally and to distribute the leftover vertical space according to the preset's
    ``crown_margin_share``. Doc 9303 3.9.1.3 asks for a centred portrait with the crown
    nearest the top edge; the precise split is not public, so it is a preset parameter
    rather than a constant invented here.

    The rectangle is returned in source coordinates and is deliberately not clipped: a
    caller needs to know that correct framing ran out of photograph. See
    :attr:`CropPlan.needs_padding`.
    """
    if head.height <= 0.0:
        msg = f"head estimate has non-positive height: {head.height}"
        raise ValueError(msg)

    crop_height = head.height / preset.target_head_height_ratio
    crop_width = crop_height * preset.aspect_ratio

    leftover = crop_height - head.height
    top = head.crown.y - leftover * preset.crown_margin_share
    left = head.eye_centre.x - crop_width / 2.0

    return CropPlan(
        rect=BoundingBox(x1=left, y1=top, x2=left + crop_width, y2=top + crop_height),
        preset=preset,
        source_size=source_size,
        head=head,
    )


def rotation_needed(head: HeadEstimate, tolerance_degrees: float) -> float:
    """Return the de-rotation angle in degrees, or 0.0 when the tilt is within tolerance.

    Doc 9303 requires cropping rather than stretching, and rotating to level the eye line
    before cropping keeps that promise: a rotation preserves the IED-to-EM ratio the
    standard uses as its stretch check.
    """
    if math.isclose(head.roll_degrees, 0.0, abs_tol=tolerance_degrees):
        return 0.0
    return -head.roll_degrees
