"""Output presets and the published requirements behind them.

Every numeric constant here is traceable to a public source, recorded in the preset's
``source`` field. That discipline exists because portrait geometry is widely misquoted:
the frequently repeated "ICAO says head height is 70-80 per cent of the photo" is a real
requirement, but Doc 9303 states it for the portrait *printed in Zone V of the document*,
not for the photograph an applicant submits.

Doc 9303 Part 3 does not itself define submitted-portrait geometry. Section 3.9.1 defers
it: portrait capturing "shall comply with relevant specifications outlined in
[ISO/IEC 39794-5]". That standard is not free, so PortraitKit implements only what is
publicly stated and marks anything beyond it as an estimate rather than a compliance
claim. See decision D009.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from portraitkit.errors import ConfigError
from portraitkit.types import ImageSize

__all__ = ["DEFAULT_PRESET", "PRESETS", "CropPreset", "get_preset", "preset_names"]

MM_PER_INCH: Final = 25.4


def _mm_to_px(millimetres: float, dpi: int) -> int:
    return round(millimetres / MM_PER_INCH * dpi)


@dataclass(frozen=True, slots=True)
class CropPreset:
    """A named output geometry.

    Attributes:
        name: Preset identifier.
        description: What the preset is for.
        output_size: Pixel dimensions of the produced crop.
        dpi: Print resolution the pixel dimensions correspond to.
        head_height_ratio: Permitted crown-to-chin extent as a fraction of output
            height, or ``None`` where no public requirement exists.
        target_head_height_ratio: What the solver aims for, normally the midpoint of the
            permitted range.
        min_interocular_mm: Minimum inter-eye distance in millimetres, or ``None``.
        centre_tolerance: Permitted horizontal offset of the face centre from the image
            centre, as a fraction of output width.
        crown_margin_share: Share of the leftover vertical space placed above the crown
            rather than below the chin. Doc 9303 asks for a centred portrait with the
            crown nearest the top edge, which any value at or below 0.5 satisfies; the
            exact figure lives in ISO/IEC 39794-5 and is not public, so this defaults to
            an even split and is left configurable per jurisdiction.
        source: Citation for every constraint above.
        makes_compliance_claim: Whether this preset targets a published standard. A
            preset that does not must never be described as compliant.
    """

    name: str
    description: str
    output_size: ImageSize
    dpi: int
    target_head_height_ratio: float
    centre_tolerance: float
    source: str
    crown_margin_share: float = 0.5
    head_height_ratio: tuple[float, float] | None = None
    min_interocular_mm: float | None = None
    makes_compliance_claim: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.target_head_height_ratio < 1.0:
            msg = (
                f"{self.name}: target_head_height_ratio must be in (0, 1), "
                f"got {self.target_head_height_ratio}"
            )
            raise ValueError(msg)
        if self.head_height_ratio is not None:
            low, high = self.head_height_ratio
            if not 0.0 < low <= high < 1.0:
                msg = f"{self.name}: head_height_ratio must satisfy 0 < min <= max < 1"
                raise ValueError(msg)
            if not low <= self.target_head_height_ratio <= high:
                msg = (
                    f"{self.name}: target_head_height_ratio "
                    f"{self.target_head_height_ratio} lies outside its own permitted "
                    f"range {self.head_height_ratio}"
                )
                raise ValueError(msg)
        if not 0.0 <= self.crown_margin_share <= 0.5:
            msg = (
                f"{self.name}: crown_margin_share must be in [0, 0.5] so the crown stays "
                f"nearest the top edge, got {self.crown_margin_share}"
            )
            raise ValueError(msg)
        if self.makes_compliance_claim and self.head_height_ratio is None:
            msg = f"{self.name}: a compliance-claiming preset must state a head-height range"
            raise ValueError(msg)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.output_size.aspect_ratio

    @property
    def min_interocular_px(self) -> float | None:
        """Minimum inter-eye distance in pixels at this preset's resolution."""
        if self.min_interocular_mm is None:
            return None
        return self.min_interocular_mm / MM_PER_INCH * self.dpi


# Doc 9303 Part 3, 3.9.1.2: submitted portraits should be 45.0 mm x 35.0 mm, and the
# width-to-height ratio of the final image has a typical value of 7:9. At 300 ppi, the
# scanning rate the same section recommends, that is 413 x 531 px.
_ICAO_DPI: Final = 300
_ICAO_WIDTH_MM: Final = 35.0
_ICAO_HEIGHT_MM: Final = 45.0

_ENTRIES: tuple[CropPreset, ...] = (
    CropPreset(
        name="icao-portrait-35x45",
        description=(
            "Travel-document portrait at the submission size Doc 9303 recommends, "
            "35 x 45 mm at 300 ppi."
        ),
        output_size=ImageSize(
            width=_mm_to_px(_ICAO_WIDTH_MM, _ICAO_DPI),
            height=_mm_to_px(_ICAO_HEIGHT_MM, _ICAO_DPI),
        ),
        dpi=_ICAO_DPI,
        head_height_ratio=(0.70, 0.80),
        target_head_height_ratio=0.75,
        min_interocular_mm=10.0,
        centre_tolerance=0.05,
        source=(
            "ICAO Doc 9303, Eighth Edition 2021, Part 3. Section 3.9.1.3 requires the "
            "printed portrait to be centred with the crown, meaning the top of the head "
            "ignoring any hair, nearest the top edge, and the crown-to-chin portion to "
            "be 70 to 80 per cent of the longest dimension of Zone V. Section 3.9.1.2 "
            "gives the 45.0 x 35.0 mm submission size, the 7:9 width-to-height ratio, "
            "and requires that modifications be made by cropping and not by stretching. "
            "Section 3.9.1.1 requires an inter-eye distance of at least 10 mm. Geometry "
            "beyond these points is delegated by 3.9.1 to ISO/IEC 39794-5, which is not "
            "publicly available."
        ),
        makes_compliance_claim=True,
    ),
    CropPreset(
        name="profile-square-512",
        description="Square portrait for a professional profile or CV. No standard applies.",
        output_size=ImageSize(width=512, height=512),
        dpi=72,
        target_head_height_ratio=0.62,
        centre_tolerance=0.08,
        source=(
            "No published standard. The framing is a conventional headshot composition "
            "chosen by this project and carries no compliance meaning."
        ),
        makes_compliance_claim=False,
    ),
)

PRESETS: Final[MappingProxyType[str, CropPreset]] = MappingProxyType(
    {entry.name: entry for entry in _ENTRIES}
)
"""Read-only preset registry."""

DEFAULT_PRESET: Final = "icao-portrait-35x45"


def preset_names() -> tuple[str, ...]:
    """Return every registered preset name."""
    return tuple(PRESETS)


def get_preset(name: str) -> CropPreset:
    """Look up a preset by name.

    Raises:
        ConfigError: If ``name`` is not registered.
    """
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(preset_names())
        msg = f"unknown crop preset {name!r}; registered presets are: {known}"
        raise ConfigError(msg) from None
