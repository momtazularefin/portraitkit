"""Head extent estimation and crop solving.

ISO/IEC 39794-5:2019 Table D.8 states its position requirements against **M**, the
midpoint of the line through the two eye centres. M is directly measurable from a
five-point landmark set, so the crop can be positioned to a measured quantity rather than
an inferred one. That is why this module solves for M's position and treats head length
as a consequence rather than the other way round.

Head length and head width are still estimates. The standard anticipates exactly this:
where crown, chin, or ears cannot be located precisely, D.1.4.4 directs that a reasoned
approximation be made. The estimates here are anchored on two measured distances the
standards themselves use, the inter-eye distance (IED) and the eye-to-mouth distance
(EM), and every value derived from them is labelled as estimated when it is assessed.

Proportions used:

* Eye line to chin is about 1.5 EM, from the mouth sitting roughly two thirds of the way
  from the eye line to the chin.
* Eye line to crown is about the same, from the eye line sitting near the vertical
  midpoint of the head.
* Head width is about twice the IED, following the note in ISO/IEC 39794-5:2019, 7.48.
"""

from __future__ import annotations

from dataclasses import dataclass

from portraitkit.crop.presets import CropPreset
from portraitkit.types import BoundingBox, FaceLandmarks5, ImageSize, Point

__all__ = [
    "CROWN_TO_EYE_PER_EM",
    "EYE_TO_CHIN_PER_EM",
    "HEAD_WIDTH_PER_IED",
    "CropPlan",
    "HeadEstimate",
    "estimate_head",
    "solve_crop",
]

EYE_TO_CHIN_PER_EM: float = 1.5
"""Eye line to chin, in units of eye-to-mouth distance."""

CROWN_TO_EYE_PER_EM: float = 1.5
"""Eye line to crown, in units of eye-to-mouth distance."""

HEAD_WIDTH_PER_IED: float = 2.0
"""Head width in units of inter-eye distance, per ISO/IEC 39794-5:2019, 7.48."""


@dataclass(frozen=True, slots=True)
class HeadEstimate:
    """Head extent inferred from landmarks.

    ``eye_centre``, ``mouth_centre``, ``interocular_distance`` and ``eye_to_mouth`` are
    measured. ``crown``, ``chin``, ``length`` and ``width`` are inferred from them.
    """

    eye_centre: Point
    """M in ISO/IEC 39794-5 Table D.8: the midpoint of the line through the eye centres."""

    mouth_centre: Point
    crown: Point
    chin: Point
    interocular_distance: float
    eye_to_mouth: float
    roll_degrees: float

    @property
    def length(self) -> float:
        """Estimated crown-to-chin extent in pixels. L in Table D.8."""
        return self.chin.y - self.crown.y

    @property
    def width(self) -> float:
        """Estimated ear-to-ear head width in pixels. W in Table D.8."""
        return HEAD_WIDTH_PER_IED * self.interocular_distance

    @property
    def ied_to_em_ratio(self) -> float:
        """The ratio ICAO Doc 9303 Part 3, 3.9.1.3 requires a resize to preserve.

        Cropping and uniform scaling both leave it unchanged, so a departure from the
        source value indicates the image was stretched.
        """
        if self.eye_to_mouth <= 0.0:
            return 0.0
        return self.interocular_distance / self.eye_to_mouth


@dataclass(frozen=True, slots=True)
class CropPlan:
    """A crop rectangle in source-image coordinates, and what realizing it will take."""

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

        A correctly framed portrait often needs canvas the original photograph does not
        contain, particularly above the crown. Reporting it lets the caller choose
        between padding and rejecting the photo, rather than silently producing a crop
        that violates the geometry it claims to satisfy.
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
    def achieved_head_length_ratio(self) -> float:
        """L/B: estimated crown-to-chin extent as a fraction of crop height."""
        return self.head.length / self.rect.height

    @property
    def achieved_head_width_ratio(self) -> float:
        """W/A: estimated head width as a fraction of crop width."""
        return self.head.width / self.rect.width

    @property
    def achieved_face_centre_vertical(self) -> float:
        """Mv/B: eye-centre midpoint measured down from the top edge."""
        return (self.head.eye_centre.y - self.rect.y1) / self.rect.height

    @property
    def achieved_face_centre_horizontal(self) -> float:
        """Mh/A: eye-centre midpoint measured across from the left edge."""
        return (self.head.eye_centre.x - self.rect.x1) / self.rect.width


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

    return HeadEstimate(
        eye_centre=eye_centre,
        mouth_centre=mouth_centre,
        crown=Point(x=eye_centre.x, y=eye_centre.y - CROWN_TO_EYE_PER_EM * eye_to_mouth),
        chin=Point(x=eye_centre.x, y=eye_centre.y + EYE_TO_CHIN_PER_EM * eye_to_mouth),
        interocular_distance=interocular,
        eye_to_mouth=eye_to_mouth,
        roll_degrees=landmarks.roll_degrees,
    )


def solve_crop(head: HeadEstimate, preset: CropPreset, source_size: ImageSize) -> CropPlan:
    """Compute the crop rectangle placing ``head`` as ``preset`` requires.

    Height is chosen so the estimated crown-to-chin extent occupies the preset's target
    L/B fraction. The rectangle is then positioned so the eye-centre midpoint M lands at
    the preset's target Mv/B down from the top edge and at the horizontal centre, which
    is what Table D.8 constrains. Width follows from the output aspect ratio.

    The rectangle is returned in source coordinates and is deliberately not clipped: a
    caller needs to know when correct framing ran out of photograph. See
    :attr:`CropPlan.needs_padding`.
    """
    if head.length <= 0.0:
        msg = f"head estimate has non-positive length: {head.length}"
        raise ValueError(msg)

    crop_height = head.length / preset.target_head_length_ratio
    crop_width = crop_height * preset.aspect_ratio

    top = head.eye_centre.y - preset.target_face_centre_vertical * crop_height
    left = head.eye_centre.x - crop_width / 2.0

    return CropPlan(
        rect=BoundingBox(x1=left, y1=top, x2=left + crop_width, y2=top + crop_height),
        preset=preset,
        source_size=source_size,
        head=head,
    )
