"""Crop presets, head estimation, crop solving, and geometry assessment.

Geometry is pinned to arithmetic. Every expected crop rectangle here is derivable by hand
from the landmark positions and the preset constants, so a change in the solver shows up
as a specific wrong number rather than a picture that still looks plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.crop.compliance import CheckBasis, CheckStatus, assess_geometry
from portraitkit.crop.geometry import (
    CROWN_TO_EYE_PER_EM,
    EYE_TO_CHIN_PER_EM,
    estimate_head,
    solve_crop,
)
from portraitkit.crop.presets import PRESETS, CropPreset, get_preset, preset_names
from portraitkit.crop.stage import CropConfig, CropStage, CropStatus
from portraitkit.errors import ConfigError
from portraitkit.types import (
    BoundingBox,
    DetectionResult,
    DetectionStatus,
    FaceDetection,
    FaceLandmarks5,
    ImageSize,
)
from tests.conftest import solid_image

# Eyes 40 px apart centred at (100, 100); mouth centre 40 px below the eye line.
LEVEL = FaceLandmarks5.from_array(
    np.asarray(
        [[80.0, 100.0], [120.0, 100.0], [100.0, 125.0], [82.0, 140.0], [118.0, 140.0]],
        dtype=np.float32,
    )
)
EM = 40.0
IED = 40.0

# The same face placed near the top edge, where the estimated crown falls outside the
# photograph and correct framing therefore needs canvas the source does not have.
NEAR_TOP = FaceLandmarks5.from_array(
    np.asarray(
        [[80.0, 45.0], [120.0, 45.0], [100.0, 70.0], [82.0, 85.0], [118.0, 85.0]],
        dtype=np.float32,
    )
)


def detection_of(landmarks: FaceLandmarks5 | None, size: ImageSize) -> DetectionResult:
    face = FaceDetection(
        box=BoundingBox(x1=60.0, y1=60.0, x2=140.0, y2=160.0), score=0.95, landmarks=landmarks
    )
    return DetectionResult(status=DetectionStatus.OK, image_size=size, faces=(face,), primary=face)


# --- presets --------------------------------------------------------------------------


def test_icao_preset_matches_the_documented_submission_size() -> None:
    """35 x 45 mm at 300 ppi, the size and rate Doc 9303 3.9.1.2 gives."""
    preset = get_preset("icao-portrait-35x45")

    assert preset.output_size == ImageSize(width=413, height=531)
    assert preset.aspect_ratio == pytest.approx(7 / 9, abs=1e-3)


def test_icao_preset_carries_the_published_head_height_range() -> None:
    preset = get_preset("icao-portrait-35x45")

    assert preset.head_height_ratio == (0.70, 0.80)
    assert preset.makes_compliance_claim


def test_minimum_interocular_distance_converts_to_pixels() -> None:
    """10 mm at 300 ppi is about 118 px."""
    assert get_preset("icao-portrait-35x45").min_interocular_px == pytest.approx(118.1, abs=0.1)


def test_every_preset_cites_a_source() -> None:
    for preset in PRESETS.values():
        assert preset.source.strip()


def test_a_preset_without_a_head_height_range_makes_no_compliance_claim() -> None:
    """A preset that cannot be checked must not be describable as compliant."""
    for preset in PRESETS.values():
        if preset.head_height_ratio is None:
            assert not preset.makes_compliance_claim


def test_profile_preset_makes_no_compliance_claim() -> None:
    assert not get_preset("profile-square-512").makes_compliance_claim


def test_unknown_preset_names_the_alternatives() -> None:
    with pytest.raises(ConfigError, match="unknown crop preset 'square'"):
        get_preset("square")


def test_preset_names_are_registered_under_their_own_name() -> None:
    for name in preset_names():
        assert PRESETS[name].name == name


def base_preset(**overrides: object) -> CropPreset:
    fields: dict[str, object] = {
        "name": "probe",
        "description": "test",
        "output_size": ImageSize(width=100, height=100),
        "dpi": 300,
        "target_head_height_ratio": 0.5,
        "centre_tolerance": 0.05,
        "source": "test",
    }
    return CropPreset(**{**fields, **overrides})  # type: ignore[arg-type]


def test_target_outside_its_own_permitted_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="lies outside its own permitted range"):
        base_preset(head_height_ratio=(0.7, 0.8), target_head_height_ratio=0.5)


def test_compliance_claim_requires_a_checkable_range() -> None:
    with pytest.raises(ValueError, match="must state a head-height range"):
        base_preset(makes_compliance_claim=True)


def test_crown_margin_share_above_half_is_rejected() -> None:
    """Above 0.5 the crown would no longer be the edge it sits nearest."""
    with pytest.raises(ValueError, match="crown_margin_share must be in"):
        base_preset(crown_margin_share=0.7)


# --- head estimation ------------------------------------------------------------------


def test_head_estimate_measures_what_it_can_measure() -> None:
    head = estimate_head(LEVEL)

    assert head.interocular_distance == pytest.approx(IED)
    assert head.eye_to_mouth == pytest.approx(EM)
    assert head.eye_centre.y == pytest.approx(100.0)


def test_crown_and_chin_follow_the_documented_proportions() -> None:
    head = estimate_head(LEVEL)

    assert head.crown.y == pytest.approx(100.0 - CROWN_TO_EYE_PER_EM * EM)
    assert head.chin.y == pytest.approx(100.0 + EYE_TO_CHIN_PER_EM * EM)
    assert head.height == pytest.approx((CROWN_TO_EYE_PER_EM + EYE_TO_CHIN_PER_EM) * EM)


def test_ied_to_em_ratio_is_scale_invariant() -> None:
    """The ratio Doc 9303 uses as its stretch check must survive uniform scaling."""
    scaled = FaceLandmarks5.from_array(
        np.asarray([[p.x * 3, p.y * 3] for p in LEVEL.as_points()], dtype=np.float32)
    )

    assert estimate_head(scaled).ied_to_em_ratio == pytest.approx(
        estimate_head(LEVEL).ied_to_em_ratio
    )


def test_coincident_eyes_are_rejected() -> None:
    degenerate = FaceLandmarks5.from_array(
        np.asarray(
            [[100.0, 100.0], [100.0, 100.0], [100.0, 125.0], [82.0, 140.0], [118.0, 140.0]],
            dtype=np.float32,
        )
    )

    with pytest.raises(ValueError, match="eye landmarks coincide"):
        estimate_head(degenerate)


def test_coincident_eye_and_mouth_are_rejected() -> None:
    degenerate = FaceLandmarks5.from_array(
        np.asarray(
            [[80.0, 100.0], [120.0, 100.0], [100.0, 100.0], [82.0, 100.0], [118.0, 100.0]],
            dtype=np.float32,
        )
    )

    with pytest.raises(ValueError, match="eye and mouth landmarks coincide"):
        estimate_head(degenerate)


# --- crop solving ---------------------------------------------------------------------


def test_crop_height_places_the_head_at_the_target_fraction() -> None:
    preset = get_preset("icao-portrait-35x45")
    head = estimate_head(LEVEL)

    plan = solve_crop(head, preset, ImageSize(width=400, height=400))

    assert plan.rect.height == pytest.approx(head.height / preset.target_head_height_ratio)
    assert plan.achieved_head_height_ratio == pytest.approx(preset.target_head_height_ratio)


def test_crop_width_follows_the_preset_aspect_ratio() -> None:
    preset = get_preset("icao-portrait-35x45")

    plan = solve_crop(estimate_head(LEVEL), preset, ImageSize(width=400, height=400))

    assert plan.rect.width / plan.rect.height == pytest.approx(preset.aspect_ratio)


def test_crop_is_centred_on_the_eye_line_horizontally() -> None:
    plan = solve_crop(
        estimate_head(LEVEL), get_preset("icao-portrait-35x45"), ImageSize(width=400, height=400)
    )

    assert plan.rect.center.x == pytest.approx(100.0)


def test_crown_margin_share_controls_vertical_placement() -> None:
    head = estimate_head(LEVEL)

    high = solve_crop(head, base_preset(crown_margin_share=0.0), ImageSize(width=400, height=400))
    even = solve_crop(head, base_preset(crown_margin_share=0.5), ImageSize(width=400, height=400))

    # With no share above, the crop starts exactly at the crown.
    assert high.rect.y1 == pytest.approx(head.crown.y)
    assert even.rect.y1 < high.rect.y1


def test_plan_reports_padding_when_framing_runs_out_of_photograph() -> None:
    """The head sits near the top, so a correct crop needs canvas above the crown."""
    plan = solve_crop(
        estimate_head(NEAR_TOP), get_preset("icao-portrait-35x45"), ImageSize(width=400, height=400)
    )

    assert plan.needs_padding
    assert plan.padding[1] > 0.0


def test_a_comfortably_placed_face_needs_no_padding() -> None:
    plan = solve_crop(
        estimate_head(LEVEL), get_preset("icao-portrait-35x45"), ImageSize(width=400, height=400)
    )

    assert not plan.needs_padding


def test_plan_reports_no_padding_with_room_to_spare() -> None:
    shifted = FaceLandmarks5.from_array(
        np.asarray([[p.x + 200, p.y + 200] for p in LEVEL.as_points()], dtype=np.float32)
    )

    plan = solve_crop(
        estimate_head(shifted), get_preset("icao-portrait-35x45"), ImageSize(width=800, height=800)
    )

    assert not plan.needs_padding
    assert plan.padding == (0.0, 0.0, 0.0, 0.0)


# --- assessment -----------------------------------------------------------------------


def roomy_plan(preset_name: str = "icao-portrait-35x45"):
    shifted = FaceLandmarks5.from_array(
        np.asarray([[p.x + 300, p.y + 300] for p in LEVEL.as_points()], dtype=np.float32)
    )
    return solve_crop(
        estimate_head(shifted), get_preset(preset_name), ImageSize(width=900, height=900)
    )


def test_a_well_framed_crop_conforms() -> None:
    assessment = assess_geometry(roomy_plan())

    assert assessment.conforms
    assert not assessment.failures


def test_head_height_check_is_labelled_as_an_estimate() -> None:
    """A pass resting on an inferred crown and chin must not read as measured fact."""
    assessment = assess_geometry(roomy_plan())

    check = next(c for c in assessment.checks if c.name == "head_height_ratio")
    assert check.basis is CheckBasis.ESTIMATED
    assert assessment.rests_on_estimates


def test_centring_and_stretch_checks_are_measured() -> None:
    assessment = assess_geometry(roomy_plan())

    for name in ("horizontal_centring", "no_stretch", "within_source_frame"):
        assert next(c for c in assessment.checks if c.name == name).basis is CheckBasis.MEASURED


def test_uniform_scaling_passes_the_stretch_check() -> None:
    """Doc 9303 3.9.1.2 requires cropping, not stretching."""
    check = next(c for c in assess_geometry(roomy_plan()).checks if c.name == "no_stretch")

    assert check.status is CheckStatus.PASS


def test_out_of_frame_framing_fails_its_check() -> None:
    plan = solve_crop(
        estimate_head(NEAR_TOP), get_preset("icao-portrait-35x45"), ImageSize(width=400, height=400)
    )

    assessment = assess_geometry(plan)

    assert not assessment.conforms
    assert [c.name for c in assessment.failures] == ["within_source_frame"]


def test_a_preset_without_a_requirement_reports_not_specified() -> None:
    assessment = assess_geometry(roomy_plan("profile-square-512"))

    check = next(c for c in assessment.checks if c.name == "head_height_ratio")
    assert check.status is CheckStatus.NOT_SPECIFIED
    assert not check.passed
    assert not assessment.makes_compliance_claim


def test_assessment_serializes() -> None:
    payload = assess_geometry(roomy_plan()).to_dict()

    assert payload["preset"] == "icao-portrait-35x45"
    assert payload["conforms"] is True
    assert {check["name"] for check in payload["checks"]} >= {
        "head_height_ratio",
        "horizontal_centring",
        "no_stretch",
        "within_source_frame",
    }


# --- stage ----------------------------------------------------------------------------


def test_stage_produces_a_preset_sized_crop() -> None:
    size = ImageSize(width=900, height=900)
    shifted = FaceLandmarks5.from_array(
        np.asarray([[p.x + 300, p.y + 300] for p in LEVEL.as_points()], dtype=np.float32)
    )
    stage = CropStage()

    result = stage.run(solid_image(900, 900), detection_of(shifted, size))

    assert result.ok
    assert result.image is not None
    assert result.image.shape == (531, 413, 3)
    assert not result.padded
    assert result.conforms


def test_stage_pads_and_says_so() -> None:
    size = ImageSize(width=400, height=400)
    stage = CropStage()

    result = stage.run(solid_image(400, 400), detection_of(NEAR_TOP, size))

    assert result.ok
    assert result.padded
    assert not result.conforms


def test_stage_can_refuse_to_pad() -> None:
    size = ImageSize(width=400, height=400)
    stage = CropStage(CropConfig(allow_padding=False))

    result = stage.run(solid_image(400, 400), detection_of(NEAR_TOP, size))

    assert result.status is CropStatus.PADDING_REQUIRED
    assert result.image is None
    assert result.assessment is not None


def test_stage_reports_no_face() -> None:
    result = CropStage().run(
        solid_image(400, 400),
        DetectionResult(status=DetectionStatus.NO_FACE, image_size=ImageSize(400, 400)),
    )

    assert result.status is CropStatus.NO_FACE
    assert not result.ok


def test_stage_reports_missing_landmarks() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_of(None, size))

    assert result.status is CropStatus.NO_LANDMARKS


def test_stage_reports_degenerate_landmarks() -> None:
    size = ImageSize(width=400, height=400)
    degenerate = FaceLandmarks5.from_array(np.full((5, 2), 100.0, dtype=np.float32))

    result = CropStage().run(solid_image(400, 400), detection_of(degenerate, size))

    assert result.status is CropStatus.DEGENERATE_LANDMARKS


def test_padding_uses_the_configured_background() -> None:
    size = ImageSize(width=400, height=400)
    stage = CropStage(CropConfig(background=(255, 0, 0)))

    result = stage.run(solid_image(400, 400, color=(10, 10, 10)), detection_of(NEAR_TOP, size))

    assert result.image is not None
    # The top rows come from padded canvas, so they carry the fill colour.
    assert result.image[0, result.image.shape[1] // 2, 0] > 200


def test_stage_records_output_metadata() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_of(LEVEL, size))

    assert result.metadata["dpi"] == 300
    assert result.metadata["output_size"] == [413, 531]
