"""Output presets and the published requirements behind them.

Every constraint here is traceable to a clause, recorded in the preset's ``source``
field. The governing table is ISO/IEC 39794-5:2019, Annex D, Table D.8, which fixes the
portrait's aspect ratio, where the face centre may sit, and how much of the frame the
head may occupy. ICAO Doc 9303 Part 3 supplies the physical submission size, and
ISO/IEC 39794-5:2019 D.1.4.2.4 supplies the inter-eye pixel counts.

Table D.8 expresses its position constraints against **M**, the midpoint of the line
through the two eye centres. That is directly measurable from a five-point landmark set,
which is why this module can state position requirements as hard checks while head width
and head length remain estimates.

Only clause references and numeric values appear here. The standards themselves are
licensed documents and are not reproduced; see `.pcp/references/20-standards-library.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from portraitkit.errors import ConfigError
from portraitkit.types import ImageSize

__all__ = ["DEFAULT_PRESET", "PRESETS", "CropPreset", "get_preset", "preset_names"]

MM_PER_INCH: Final = 25.4

# ISO/IEC 39794-5:2019, D.1.4.4 Table D.8.
TABLE_D8_ASPECT_RATIO: Final = (0.74, 0.80)
TABLE_D8_FACE_CENTRE_HORIZONTAL: Final = (0.45, 0.55)
TABLE_D8_FACE_CENTRE_VERTICAL: Final = (0.30, 0.50)
TABLE_D8_HEAD_WIDTH_RATIO: Final = (0.50, 0.75)
TABLE_D8_HEAD_LENGTH_RATIO: Final = (0.60, 0.90)

# ISO/IEC 39794-5:2019, D.1.4.5.4 relaxes two of the above for children up to eleven.
CHILD_HEAD_LENGTH_RATIO: Final = (0.50, 0.90)
CHILD_FACE_CENTRE_VERTICAL: Final = (0.30, 0.60)

# ISO/IEC 39794-5:2019, D.1.4.2.4 inter-eye pixel counts.
IED_LEGACY_PX: Final = 90
IED_NEW_PASSPORT_PX: Final = 240


def _mm_to_px(millimetres: float, dpi: int) -> int:
    return round(millimetres / MM_PER_INCH * dpi)


def _midpoint(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


@dataclass(frozen=True, slots=True)
class CropPreset:
    """A named output geometry.

    Ranges are inclusive bounds as fractions. ``target_*`` values are what the solver
    aims for and must lie inside their corresponding range.
    """

    name: str
    description: str
    output_size: ImageSize
    dpi: int
    source: str

    head_length_ratio: tuple[float, float] | None = None
    """Permitted crown-to-chin extent as a fraction of image height (L/B)."""

    target_head_length_ratio: float = 0.75
    """What the solver aims for within :attr:`head_length_ratio`."""

    face_centre_vertical: tuple[float, float] | None = None
    """Permitted position of the eye-centre midpoint from the top edge (Mv/B)."""

    target_face_centre_vertical: float = 0.40
    """What the solver aims for within :attr:`face_centre_vertical`."""

    face_centre_horizontal: tuple[float, float] | None = None
    """Permitted position of the eye-centre midpoint from the left edge (Mh/A)."""

    head_width_ratio: tuple[float, float] | None = None
    """Permitted head width as a fraction of image width (W/A)."""

    aspect_ratio_range: tuple[float, float] | None = None
    """Permitted image width to height ratio (A/B)."""

    min_interocular_px: float | None = None
    """Minimum inter-eye distance in the produced output, in pixels."""

    makes_compliance_claim: bool = False
    """Whether this preset targets a published standard. A preset that does not must
    never be described as compliant."""

    def __post_init__(self) -> None:
        for label, bounds in (
            ("head_length_ratio", self.head_length_ratio),
            ("face_centre_vertical", self.face_centre_vertical),
            ("face_centre_horizontal", self.face_centre_horizontal),
            ("head_width_ratio", self.head_width_ratio),
            ("aspect_ratio_range", self.aspect_ratio_range),
        ):
            if bounds is not None and not 0.0 < bounds[0] <= bounds[1] < 1.0:
                msg = f"{self.name}: {label} must satisfy 0 < min <= max < 1, got {bounds}"
                raise ValueError(msg)

        for label, target, bounds in (
            ("target_head_length_ratio", self.target_head_length_ratio, self.head_length_ratio),
            (
                "target_face_centre_vertical",
                self.target_face_centre_vertical,
                self.face_centre_vertical,
            ),
        ):
            if not 0.0 < target < 1.0:
                msg = f"{self.name}: {label} must be in (0, 1), got {target}"
                raise ValueError(msg)
            if bounds is not None and not bounds[0] <= target <= bounds[1]:
                msg = (
                    f"{self.name}: {label} of {target} lies outside its own permitted "
                    f"range {bounds}"
                )
                raise ValueError(msg)

        if self.makes_compliance_claim and self.head_length_ratio is None:
            msg = f"{self.name}: a compliance-claiming preset must state a head-length range"
            raise ValueError(msg)

    @property
    def aspect_ratio(self) -> float:
        """Actual width divided by height of the produced output."""
        return self.output_size.aspect_ratio


_DOC_9303_SIZE: Final = "ICAO Doc 9303 Part 3, 3.9.1.2 (45.0 x 35.0 mm submission size)"
_TABLE_D8: Final = (
    "ISO/IEC 39794-5:2019, D.1.4.4 and Table D.8 (A/B, Mh/A, Mv/B, W/A, L/B); "
    "D.1.4.2.4 (inter-eye pixel count)"
)
_WIDTH_MM: Final = 35.0
_HEIGHT_MM: Final = 45.0


def _icao_preset(name: str, dpi: int, min_ied: int, note: str) -> CropPreset:
    return CropPreset(
        name=name,
        description=f"Travel-document portrait, 35 x 45 mm at {dpi} ppi. {note}",
        output_size=ImageSize(width=_mm_to_px(_WIDTH_MM, dpi), height=_mm_to_px(_HEIGHT_MM, dpi)),
        dpi=dpi,
        head_length_ratio=TABLE_D8_HEAD_LENGTH_RATIO,
        target_head_length_ratio=0.75,
        face_centre_vertical=TABLE_D8_FACE_CENTRE_VERTICAL,
        target_face_centre_vertical=_midpoint(TABLE_D8_FACE_CENTRE_VERTICAL),
        face_centre_horizontal=TABLE_D8_FACE_CENTRE_HORIZONTAL,
        head_width_ratio=TABLE_D8_HEAD_WIDTH_RATIO,
        aspect_ratio_range=TABLE_D8_ASPECT_RATIO,
        min_interocular_px=float(min_ied),
        source=f"{_TABLE_D8}. {_DOC_9303_SIZE}.",
        makes_compliance_claim=True,
    )


_ENTRIES: tuple[CropPreset, ...] = (
    _icao_preset(
        "icao-portrait-35x45",
        dpi=300,
        min_ied=IED_LEGACY_PX,
        note=(
            "Meets the legacy inter-eye pixel count. A 35 mm frame at 300 ppi cannot "
            "reach the count required of new passport application processes; use the "
            "600 ppi preset for those."
        ),
    ),
    _icao_preset(
        "icao-portrait-35x45-600",
        dpi=600,
        min_ied=IED_NEW_PASSPORT_PX,
        note="Meets the inter-eye pixel count required of new passport application processes.",
    ),
    CropPreset(
        name="profile-square-512",
        description="Square portrait for a professional profile or CV. No standard applies.",
        output_size=ImageSize(width=512, height=512),
        dpi=72,
        target_head_length_ratio=0.62,
        target_face_centre_vertical=0.42,
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
