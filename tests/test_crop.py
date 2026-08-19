"""Crop presets, head estimation, crop solving, and geometry assessment.

Geometry is pinned to arithmetic. Every expected rectangle is derivable by hand from the
landmark positions and the preset constants, so a change in the solver shows up as a
specific wrong number rather than a picture that still looks plausible.

Requirement values are checked against ISO/IEC 39794-5:2019 Table D.8 by clause
reference. The standards are licensed and are not reproduced here.
"""

from __future__ import annotations

import numpy as np
import pytest

from portraitkit.crop.compliance import CheckBasis, CheckStatus, assess_geometry
from portraitkit.crop.geometry import (
    CROWN_TO_EYE_PER_EM,
    EYE_TO_CHIN_PER_EM,
    HEAD_WIDTH_PER_IED,
    estimate_head,
    solve_crop,
)
from portraitkit.crop.presets import (
    PRESETS,
    TABLE_D8_ASPECT_RATIO,
    TABLE_D8_FACE_CENTRE_HORIZONTAL,
    TABLE_D8_FACE_CENTRE_VERTICAL,
    TABLE_D8_HEAD_LENGTH_RATIO,
    TABLE_D8_HEAD_WIDTH_RATIO,
    CropPreset,
    get_preset,
    preset_names,
)
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

# The same face near the top edge, where correct framing needs canvas above the crown.
NEAR_TOP = FaceLandmarks5.from_array(
    np.asarray(
        [[80.0, 45.0], [120.0, 45.0], [100.0, 70.0], [82.0, 85.0], [118.0, 85.0]],
        dtype=np.float32,
    )
)

ICAO = "icao-portrait-35x45"


def detection_of(landmarks: FaceLandmarks5 | None, size: ImageSize) -> DetectionResult:
    face = FaceDetection(
        box=BoundingBox(x1=60.0, y1=60.0, x2=140.0, y2=160.0), score=0.95, landmarks=landmarks
    )
    return DetectionResult(status=DetectionStatus.OK, image_size=size, faces=(face,), primary=face)


def base_preset(**overrides: object) -> CropPreset:
    fields: dict[str, object] = {
        "name": "probe",
        "description": "test",
        "output_size": ImageSize(width=100, height=100),
        "dpi": 300,
        "source": "test",
    }
    return CropPreset(**{**fields, **overrides})  # type: ignore[arg-type]


# --- presets --------------------------------------------------------------------------


def test_icao_preset_matches_the_documented_submission_size() -> None:
    """35 x 45 mm at 300 ppi, the size ICAO Doc 9303 Part 3, 3.9.1.2 gives."""
    assert get_preset(ICAO).output_size == ImageSize(width=413, height=531)


@pytest.mark.parametrize("name", ["icao-portrait-35x45", "icao-portrait-35x45-600"])
def test_icao_presets_sit_inside_the_permitted_aspect_ratio(name: str) -> None:
    """Table D.8 constrains A/B; a preset violating its own table would be incoherent."""
    preset = get_preset(name)

    low, high = TABLE_D8_ASPECT_RATIO
    assert low <= preset.aspect_ratio <= high


def test_icao_presets_carry_the_table_d8_ranges() -> None:
    preset = get_preset(ICAO)

    assert preset.head_length_ratio == TABLE_D8_HEAD_LENGTH_RATIO
    assert preset.head_width_ratio == TABLE_D8_HEAD_WIDTH_RATIO
    assert preset.face_centre_vertical == TABLE_D8_FACE_CENTRE_VERTICAL
    assert preset.face_centre_horizontal == TABLE_D8_FACE_CENTRE_HORIZONTAL
    assert preset.makes_compliance_claim


def test_the_two_inter_eye_pixel_counts_are_distinguished() -> None:
    """D.1.4.2.4 sets one count for legacy use and a higher one for new processes."""
    assert get_preset(ICAO).min_interocular_px == 90.0
    assert get_preset("icao-portrait-35x45-600").min_interocular_px == 240.0


def test_the_higher_count_needs_the_higher_resolution_preset() -> None:
    """A 35 mm frame at 300 ppi cannot reach the new-process count, which is why the
    600 ppi preset exists rather than a stricter threshold on the same output size."""
    legacy = get_preset(ICAO)
    modern = get_preset("icao-portrait-35x45-600")

    assert modern.min_interocular_px > legacy.output_size.width * 0.5
    assert modern.min_interocular_px < modern.output_size.width * 0.5


def test_every_preset_cites_a_source() -> None:
    for preset in PRESETS.values():
        assert preset.source.strip()


def test_a_preset_without_a_head_length_range_makes_no_compliance_claim() -> None:
    """A preset that cannot be checked must not be describable as compliant."""
    for preset in PRESETS.values():
        if preset.head_length_ratio is None:
            assert not preset.makes_compliance_claim


def test_unknown_preset_names_the_alternatives() -> None:
    with pytest.raises(ConfigError, match="unknown crop preset 'square'"):
        get_preset("square")


def test_preset_names_are_registered_under_their_own_name() -> None:
    for name in preset_names():
        assert PRESETS[name].name == name


def test_target_outside_its_own_permitted_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="lies outside its own permitted range"):
        base_preset(head_length_ratio=(0.6, 0.9), target_head_length_ratio=0.95)


def test_inverted_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="must satisfy 0 < min <= max < 1"):
        base_preset(head_length_ratio=(0.9, 0.6))


def test_compliance_claim_requires_a_checkable_range() -> None:
    with pytest.raises(ValueError, match="must state a head-length range"):
        base_preset(makes_compliance_claim=True)


# --- head estimation ------------------------------------------------------------------


def test_head_estimate_measures_what_it_can_measure() -> None:
    head = estimate_head(LEVEL)

    assert head.interocular_distance == pytest.approx(IED)
    assert head.eye_to_mouth == pytest.approx(EM)
    # M in Table D.8 is the midpoint of the line through the eye centres.
    assert head.eye_centre.y == pytest.approx(100.0)
    assert head.eye_centre.x == pytest.approx(100.0)


def test_crown_and_chin_follow_the_documented_proportions() -> None:
    head = estimate_head(LEVEL)

    assert head.crown.y == pytest.approx(100.0 - CROWN_TO_EYE_PER_EM * EM)
    assert head.chin.y == pytest.approx(100.0 + EYE_TO_CHIN_PER_EM * EM)
    assert head.length == pytest.approx((CROWN_TO_EYE_PER_EM + EYE_TO_CHIN_PER_EM) * EM)


def test_head_width_follows_the_inter_eye_relation() -> None:
    """ISO/IEC 39794-5:2019, 7.48 notes the typical IED is about half the head width."""
    assert estimate_head(LEVEL).width == pytest.approx(HEAD_WIDTH_PER_IED * IED)


def test_ied_to_em_ratio_is_scale_invariant() -> None:
    """The ratio Doc 9303 uses as its stretch check must survive uniform scaling."""
    scaled = FaceLandmarks5.from_array(
        np.asarray([[p.x * 3, p.y * 3] for p in LEVEL.as_points()], dtype=np.float32)
    )

    assert estimate_head(scaled).ied_to_em_ratio == pytest.approx(
        estimate_head(LEVEL).ied_to_em_ratio
    )


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (
            [[100.0, 100.0], [100.0, 100.0], [100.0, 125.0], [82.0, 140.0], [118.0, 140.0]],
            "eye landmarks coincide",
        ),
        (
            [[80.0, 100.0], [120.0, 100.0], [100.0, 100.0], [82.0, 100.0], [118.0, 100.0]],
            "eye and mouth landmarks coincide",
        ),
    ],
)
def test_degenerate_landmarks_are_rejected(points: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_head(FaceLandmarks5.from_array(np.asarray(points, dtype=np.float32)))


# --- crop solving ---------------------------------------------------------------------


def test_crop_height_places_the_head_at_the_target_fraction() -> None:
    preset = get_preset(ICAO)
    head = estimate_head(LEVEL)

    plan = solve_crop(head, preset, ImageSize(width=400, height=400))

    assert plan.rect.height == pytest.approx(head.length / preset.target_head_length_ratio)
    assert plan.achieved_head_length_ratio == pytest.approx(preset.target_head_length_ratio)


def test_crop_places_the_face_centre_at_the_target_position() -> None:
    """Mv/B and Mh/A are what Table D.8 constrains, and both are measured quantities."""
    preset = get_preset(ICAO)

    plan = solve_crop(estimate_head(LEVEL), preset, ImageSize(width=400, height=400))

    assert plan.achieved_face_centre_vertical == pytest.approx(preset.target_face_centre_vertical)
    assert plan.achieved_face_centre_horizontal == pytest.approx(0.5)


def test_crop_geometry_matches_hand_computed_values() -> None:
    """Head length 120, target 0.75, so height 160; M at 0.40 down puts the top at 36."""
    plan = solve_crop(estimate_head(LEVEL), get_preset(ICAO), ImageSize(width=400, height=400))

    assert plan.rect.height == pytest.approx(160.0)
    assert plan.rect.y1 == pytest.approx(36.0)
    assert plan.rect.width == pytest.approx(160.0 * get_preset(ICAO).aspect_ratio)
    assert plan.rect.center.x == pytest.approx(100.0)


def test_crop_width_follows_the_preset_aspect_ratio() -> None:
    preset = get_preset(ICAO)

    plan = solve_crop(estimate_head(LEVEL), preset, ImageSize(width=400, height=400))

    assert plan.rect.width / plan.rect.height == pytest.approx(preset.aspect_ratio)


def test_solved_crop_satisfies_every_table_d8_range() -> None:
    """The targets must be mutually consistent, not merely individually reasonable."""
    plan = solve_crop(estimate_head(LEVEL), get_preset(ICAO), ImageSize(width=400, height=400))

    for value, bounds in (
        (plan.achieved_head_length_ratio, TABLE_D8_HEAD_LENGTH_RATIO),
        (plan.achieved_head_width_ratio, TABLE_D8_HEAD_WIDTH_RATIO),
        (plan.achieved_face_centre_vertical, TABLE_D8_FACE_CENTRE_VERTICAL),
        (plan.achieved_face_centre_horizontal, TABLE_D8_FACE_CENTRE_HORIZONTAL),
    ):
        assert bounds[0] <= value <= bounds[1]


def test_a_comfortably_placed_face_needs_no_padding() -> None:
    plan = solve_crop(estimate_head(LEVEL), get_preset(ICAO), ImageSize(width=400, height=400))

    assert not plan.needs_padding
    assert plan.padding == (0.0, 0.0, 0.0, 0.0)


def test_plan_reports_padding_when_framing_runs_out_of_photograph() -> None:
    plan = solve_crop(estimate_head(NEAR_TOP), get_preset(ICAO), ImageSize(width=400, height=400))

    assert plan.needs_padding
    assert plan.padding[1] > 0.0


# --- assessment -----------------------------------------------------------------------


def plan_for(landmarks: FaceLandmarks5 = LEVEL, preset_name: str = ICAO, size: int = 400):
    return solve_crop(
        estimate_head(landmarks), get_preset(preset_name), ImageSize(width=size, height=size)
    )


def test_a_well_framed_crop_conforms() -> None:
    assessment = assess_geometry(plan_for())

    assert assessment.conforms
    assert not assessment.failures


def test_every_table_d8_row_is_checked() -> None:
    names = {check.name for check in assess_geometry(plan_for()).checks}

    assert names >= {
        "head_length_ratio",
        "head_width_ratio",
        "face_centre_vertical",
        "face_centre_horizontal",
        "aspect_ratio",
        "interocular_distance_px",
    }


def test_position_checks_are_measured_and_extent_checks_are_estimated() -> None:
    """M comes from landmarks; crown, chin, and ears do not."""
    checks = {check.name: check for check in assess_geometry(plan_for()).checks}

    assert checks["face_centre_vertical"].basis is CheckBasis.MEASURED
    assert checks["face_centre_horizontal"].basis is CheckBasis.MEASURED
    assert checks["head_length_ratio"].basis is CheckBasis.ESTIMATED
    assert checks["head_width_ratio"].basis is CheckBasis.ESTIMATED


def test_estimated_checks_are_reported_as_such() -> None:
    assert assess_geometry(plan_for()).rests_on_estimates


def test_checks_carry_their_clause() -> None:
    """A reviewer holding the standard must be able to verify each number."""
    checks = {check.name: check for check in assess_geometry(plan_for()).checks}

    assert "Table D.8" in checks["head_length_ratio"].clause
    assert "D.1.4.2.4" in checks["interocular_distance_px"].clause


def test_uniform_scaling_passes_the_stretch_check() -> None:
    check = next(c for c in assess_geometry(plan_for()).checks if c.name == "no_stretch")

    assert check.status is CheckStatus.PASS


def test_inter_eye_pixel_count_is_met_by_both_icao_presets() -> None:
    for name in ("icao-portrait-35x45", "icao-portrait-35x45-600"):
        check = next(
            c
            for c in assess_geometry(plan_for(preset_name=name)).checks
            if c.name == "interocular_distance_px"
        )
        assert check.status is CheckStatus.PASS, name


def test_out_of_frame_framing_fails_its_check() -> None:
    assessment = assess_geometry(plan_for(NEAR_TOP))

    assert not assessment.conforms
    assert [check.name for check in assessment.failures] == ["within_source_frame"]


def test_a_preset_without_a_requirement_reports_not_specified() -> None:
    assessment = assess_geometry(plan_for(preset_name="profile-square-512"))

    check = next(c for c in assessment.checks if c.name == "head_length_ratio")
    assert check.status is CheckStatus.NOT_SPECIFIED
    assert not check.passed
    assert not assessment.makes_compliance_claim


def test_assessment_serializes_with_clauses() -> None:
    payload = assess_geometry(plan_for()).to_dict()

    assert payload["preset"] == ICAO
    assert payload["conforms"] is True
    assert payload["rests_on_estimates"] is True
    head_length = next(c for c in payload["checks"] if c["name"] == "head_length_ratio")
    assert head_length["clause"]
    assert head_length["permitted"] == [0.60, 0.90]


# --- stage ----------------------------------------------------------------------------


def test_stage_produces_a_preset_sized_crop() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_of(LEVEL, size))

    assert result.ok
    assert result.image is not None
    assert result.image.shape == (531, 413, 3)
    assert not result.padded
    assert result.conforms


def test_stage_honours_the_selected_preset() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage(CropConfig(preset="icao-portrait-35x45-600")).run(
        solid_image(400, 400), detection_of(LEVEL, size)
    )

    assert result.image is not None
    assert result.image.shape == (1063, 827, 3)


def test_stage_pads_and_says_so() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_of(NEAR_TOP, size))

    assert result.ok
    assert result.padded
    assert not result.conforms


def test_stage_can_refuse_to_pad() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage(CropConfig(allow_padding=False)).run(
        solid_image(400, 400), detection_of(NEAR_TOP, size)
    )

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
    assert result.image[0, result.image.shape[1] // 2, 0] > 200


def test_stage_records_output_metadata() -> None:
    size = ImageSize(width=400, height=400)

    result = CropStage().run(solid_image(400, 400), detection_of(LEVEL, size))

    assert result.metadata["dpi"] == 300
    assert result.metadata["output_size"] == [413, 531]
    assert result.metadata["head_length_ratio"] == pytest.approx(0.75)
    assert result.metadata["face_centre_vertical"] == pytest.approx(0.40)
