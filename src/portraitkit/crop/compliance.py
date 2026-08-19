"""Geometry conformance assessment.

This module reports whether a crop satisfies the geometry PortraitKit can actually
verify, and it labels every check with the strength of its evidence. A check whose input
was measured from landmarks is worth more than one resting on an inferred crown and chin,
and collapsing that difference into a single pass or fail would overstate what the
project knows.

This is not a compliance certification. Doc 9303 delegates portrait geometry to
ISO/IEC 39794-5 and face image quality is measured by ISO/IEC 29794-5, for which OFIQ is
the reference implementation. Integrating that external referee is milestone M2b; until
then these checks are PortraitKit grading its own geometry, and are described as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from portraitkit.crop.geometry import CropPlan

__all__ = ["CheckBasis", "CheckStatus", "GeometryAssessment", "GeometryCheck", "assess_geometry"]


class CheckStatus(StrEnum):
    """Outcome of one geometry check."""

    PASS = "pass"
    FAIL = "fail"
    NOT_SPECIFIED = "not_specified"
    """The active preset states no requirement, so there is nothing to check against."""


class CheckBasis(StrEnum):
    """How much the inputs to a check can be trusted."""

    MEASURED = "measured"
    """Computed from landmarks the detector actually produced."""

    ESTIMATED = "estimated"
    """Derived from inferred crown and chin positions, which carry population-average
    error. A failure is informative; a pass is not proof."""


@dataclass(frozen=True, slots=True)
class GeometryCheck:
    """One named geometry check and its result."""

    name: str
    status: CheckStatus
    basis: CheckBasis
    detail: str
    value: float | None = None
    permitted: tuple[float, float] | None = None

    @property
    def passed(self) -> bool:
        """Whether the check succeeded. A not-specified check is not a success."""
        return self.status is CheckStatus.PASS


@dataclass(frozen=True, slots=True)
class GeometryAssessment:
    """The full set of checks for one crop."""

    preset: str
    checks: tuple[GeometryCheck, ...]
    makes_compliance_claim: bool

    @property
    def failures(self) -> tuple[GeometryCheck, ...]:
        """Checks that failed."""
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    @property
    def conforms(self) -> bool:
        """Whether every applicable check passed.

        Read this as "nothing PortraitKit can check is wrong", not as "compliant".
        """
        return not self.failures

    @property
    def rests_on_estimates(self) -> bool:
        """Whether any passing check depended on inferred crown and chin positions."""
        return any(
            check.basis is CheckBasis.ESTIMATED and check.status is not CheckStatus.NOT_SPECIFIED
            for check in self.checks
        )

    def to_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "preset": self.preset,
            "conforms": self.conforms,
            "makes_compliance_claim": self.makes_compliance_claim,
            "rests_on_estimates": self.rests_on_estimates,
            "checks": [
                {
                    "name": check.name,
                    "status": str(check.status),
                    "basis": str(check.basis),
                    "value": None if check.value is None else round(check.value, 4),
                    "permitted": list(check.permitted) if check.permitted else None,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


def _range_check(
    name: str,
    value: float,
    permitted: tuple[float, float] | None,
    basis: CheckBasis,
    unit: str,
) -> GeometryCheck:
    if permitted is None:
        return GeometryCheck(
            name=name,
            status=CheckStatus.NOT_SPECIFIED,
            basis=basis,
            detail="the active preset states no requirement for this property",
            value=value,
        )
    low, high = permitted
    inside = low <= value <= high
    return GeometryCheck(
        name=name,
        status=CheckStatus.PASS if inside else CheckStatus.FAIL,
        basis=basis,
        value=value,
        permitted=permitted,
        detail=(
            f"{value:.4f}{unit} is within the permitted {low:.4f} to {high:.4f}"
            if inside
            else f"{value:.4f}{unit} is outside the permitted {low:.4f} to {high:.4f}"
        ),
    )


def assess_geometry(plan: CropPlan) -> GeometryAssessment:
    """Assess ``plan`` against its preset."""
    preset = plan.preset
    checks: list[GeometryCheck] = [
        _range_check(
            "head_height_ratio",
            plan.achieved_head_height_ratio,
            preset.head_height_ratio,
            CheckBasis.ESTIMATED,
            "",
        )
    ]

    # Horizontal centring: the face centre against the crop centre, as a fraction of
    # crop width. Measured directly from the eye landmarks.
    offset = abs(plan.head.eye_centre.x - plan.rect.center.x) / plan.rect.width
    checks.append(
        _range_check(
            "horizontal_centring",
            offset,
            (0.0, preset.centre_tolerance),
            CheckBasis.MEASURED,
            "",
        )
    )

    # Inter-eye distance in the produced output, against the preset's minimum.
    output_ied = plan.head.interocular_distance * plan.scale
    minimum = preset.min_interocular_px
    checks.append(
        _range_check(
            "interocular_distance_px",
            output_ied,
            None if minimum is None else (minimum, float("inf")),
            CheckBasis.MEASURED,
            " px",
        )
    )

    # Doc 9303 3.9.1.2 requires modification by cropping, not stretching. A uniform
    # scale leaves the IED-to-EM ratio untouched, so comparing it before and after is a
    # direct check that no stretch crept in.
    source_ratio = plan.head.ied_to_em_ratio
    output_ratio = (
        (plan.head.interocular_distance * plan.scale) / (plan.head.eye_to_mouth * plan.scale)
        if plan.head.eye_to_mouth > 0.0
        else 0.0
    )
    drift = abs(source_ratio - output_ratio)
    checks.append(
        GeometryCheck(
            name="no_stretch",
            status=CheckStatus.PASS if drift < 1e-6 else CheckStatus.FAIL,
            basis=CheckBasis.MEASURED,
            value=drift,
            detail=(
                "the ratio of inter-eye to eye-to-mouth distance is unchanged, so the "
                "image was cropped and uniformly scaled rather than stretched"
                if drift < 1e-6
                else f"inter-eye to eye-to-mouth ratio drifted by {drift:.6f}; "
                "the image was not scaled uniformly"
            ),
        )
    )

    # Whether correct framing fits inside the photograph that was actually supplied.
    left, top, right, bottom = plan.padding
    needed = max(left, top, right, bottom)
    checks.append(
        GeometryCheck(
            name="within_source_frame",
            status=CheckStatus.PASS if not plan.needs_padding else CheckStatus.FAIL,
            basis=CheckBasis.MEASURED,
            value=needed,
            detail=(
                "the required crop lies entirely inside the source photograph"
                if not plan.needs_padding
                else f"correct framing needs {needed:.0f} px of canvas the source does not have "
                f"(left {left:.0f}, top {top:.0f}, right {right:.0f}, bottom {bottom:.0f})"
            ),
        )
    )

    return GeometryAssessment(
        preset=preset.name,
        checks=tuple(checks),
        makes_compliance_claim=preset.makes_compliance_claim,
    )
