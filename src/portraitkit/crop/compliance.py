"""Geometry conformance assessment.

Checks the produced crop against the constraints its preset declares, and labels every
check with the strength of its evidence. A check whose inputs were measured from
landmarks is worth more than one resting on an inferred crown, chin, or ear position, and
collapsing that difference into a single pass or fail would overstate what is known.

Position checks against ISO/IEC 39794-5:2019 Table D.8 are measured, because the table
constrains the eye-centre midpoint M and M comes straight from the landmarks. Head length
and head width are estimated. The standard expects that: D.1.4.4 directs a reasoned
approximation where crown, chin, or ears cannot be located precisely.

This is not a compliance certification. Face image quality is measured by ISO/IEC
29794-5, for which pinned OFIQ 1.0.3 is the external reference implementation. OFIQ now
supplies a separate quality evaluation twin, but these checks remain PortraitKit-authored
geometry evidence and must not be presented as external certification.
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
    """Derived from inferred crown, chin, or ear positions, which carry
    population-average error. A failure is informative; a pass is not proof."""


@dataclass(frozen=True, slots=True)
class GeometryCheck:
    """One named geometry check and its result."""

    name: str
    status: CheckStatus
    basis: CheckBasis
    detail: str
    value: float | None = None
    permitted: tuple[float, float] | None = None
    clause: str = ""
    """The clause the requirement comes from, for a reviewer holding the standard."""

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
        """Whether any applicable check depended on an inferred head boundary."""
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
                    "clause": check.clause,
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
    clause: str,
    unit: str = "",
) -> GeometryCheck:
    if permitted is None:
        return GeometryCheck(
            name=name,
            status=CheckStatus.NOT_SPECIFIED,
            basis=basis,
            detail="the active preset states no requirement for this property",
            value=value,
            clause=clause,
        )
    low, high = permitted
    inside = low <= value <= high
    high_text = "no upper bound" if high == float("inf") else f"{high:.4f}"
    return GeometryCheck(
        name=name,
        status=CheckStatus.PASS if inside else CheckStatus.FAIL,
        basis=basis,
        value=value,
        permitted=permitted,
        clause=clause,
        detail=(
            f"{value:.4f}{unit} is within the permitted {low:.4f} to {high_text}"
            if inside
            else f"{value:.4f}{unit} is outside the permitted {low:.4f} to {high_text}"
        ),
    )


_TABLE_D8 = "ISO/IEC 39794-5:2019, Table D.8"
_IED_CLAUSE = "ISO/IEC 39794-5:2019, D.1.4.2.4"
_STRETCH_CLAUSE = "ICAO Doc 9303 Part 3, 3.9.1.2"


def assess_geometry(plan: CropPlan) -> GeometryAssessment:
    """Assess ``plan`` against its preset."""
    preset = plan.preset
    checks: list[GeometryCheck] = [
        # L/B and W/A depend on inferred head boundaries.
        _range_check(
            "head_length_ratio",
            plan.achieved_head_length_ratio,
            preset.head_length_ratio,
            CheckBasis.ESTIMATED,
            _TABLE_D8,
        ),
        _range_check(
            "head_width_ratio",
            plan.achieved_head_width_ratio,
            preset.head_width_ratio,
            CheckBasis.ESTIMATED,
            _TABLE_D8,
        ),
        # Mv/B and Mh/A constrain the eye-centre midpoint, which is measured.
        _range_check(
            "face_centre_vertical",
            plan.achieved_face_centre_vertical,
            preset.face_centre_vertical,
            CheckBasis.MEASURED,
            _TABLE_D8,
        ),
        _range_check(
            "face_centre_horizontal",
            plan.achieved_face_centre_horizontal,
            preset.face_centre_horizontal,
            CheckBasis.MEASURED,
            _TABLE_D8,
        ),
        _range_check(
            "aspect_ratio",
            preset.aspect_ratio,
            preset.aspect_ratio_range,
            CheckBasis.MEASURED,
            _TABLE_D8,
        ),
        _range_check(
            "interocular_distance_px",
            plan.head.interocular_distance * plan.scale,
            None
            if preset.min_interocular_px is None
            else (preset.min_interocular_px, float("inf")),
            CheckBasis.MEASURED,
            _IED_CLAUSE,
            unit=" px",
        ),
    ]

    # A uniform scale leaves the IED-to-EM ratio untouched, so comparing it before and
    # after is a direct check that no stretch crept in.
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
            clause=_STRETCH_CLAUSE,
            detail=(
                "the ratio of inter-eye to eye-to-mouth distance is unchanged, so the "
                "image was cropped and uniformly scaled rather than stretched"
                if drift < 1e-6
                else f"inter-eye to eye-to-mouth ratio drifted by {drift:.6f}; "
                "the image was not scaled uniformly"
            ),
        )
    )

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
                else f"correct framing needs {needed:.0f} px of canvas the source does not "
                f"have (left {left:.0f}, top {top:.0f}, right {right:.0f}, bottom {bottom:.0f})"
            ),
        )
    )

    return GeometryAssessment(
        preset=preset.name,
        checks=tuple(checks),
        makes_compliance_claim=preset.makes_compliance_claim,
    )
