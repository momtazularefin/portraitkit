"""Primary-subject selection and the detection stage.

The stage is exercised through a stub detector so its diagnosis, status, and ordering
logic are covered without a model file. Tests that need real weights live at the bottom
and skip themselves when the artifact is not already cached, which keeps CI offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from portraitkit.config import load_settings
from portraitkit.detection.base import DetectorConfig
from portraitkit.detection.selection import SelectionStrategy, select_primary
from portraitkit.detection.stage import DetectionStage, StageConfig, build_detector
from portraitkit.errors import ModelError
from portraitkit.imaging.io import LoadedImage
from portraitkit.imaging.orientation import ExifOrientation, OrientationFix
from portraitkit.models.registry import DEFAULT_DETECTOR
from portraitkit.models.session import SessionInfo
from portraitkit.models.store import is_cached
from portraitkit.types import (
    BoundingBox,
    DetectionStatus,
    Diagnostic,
    FaceDetection,
    FaceLandmarks5,
    ImageSize,
)
from tests.conftest import solid_image

FRAME = ImageSize(width=200, height=200)


def face(
    x1: float, y1: float, x2: float, y2: float, score: float = 0.9, roll: float = 0.0
) -> FaceDetection:
    landmarks = None
    if roll:
        offset = np.tan(np.radians(roll)) * 20.0
        landmarks = FaceLandmarks5.from_array(
            np.asarray(
                [
                    [x1 + 10.0, y1 + 10.0],
                    [x1 + 30.0, y1 + 10.0 + offset],
                    [x1 + 20.0, y1 + 20.0],
                    [x1 + 12.0, y1 + 30.0],
                    [x1 + 28.0, y1 + 30.0],
                ],
                dtype=np.float32,
            )
        )
    return FaceDetection(
        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), score=score, landmarks=landmarks
    )


class StubDetector:
    """Stands in for an adapter so stage logic is testable without a model."""

    def __init__(
        self,
        faces: tuple[FaceDetection, ...],
        name: str = "stub",
        config: DetectorConfig | None = None,
    ) -> None:
        self._faces = faces
        self.name = name
        self.config = config or DetectorConfig()
        self.info = SessionInfo(inputs=(), outputs=(), providers=("CPUExecutionProvider",))

    def detect(self, image_rgb: np.ndarray) -> tuple[FaceDetection, ...]:
        del image_rgb
        return self._faces


def loaded(
    pixels: np.ndarray,
    *,
    orientation: ExifOrientation = ExifOrientation.TOP_LEFT,
    truncated: bool = False,
) -> LoadedImage:
    size = ImageSize(width=int(pixels.shape[1]), height=int(pixels.shape[0]))
    return LoadedImage(
        pixels=pixels,
        orientation=OrientationFix(
            original=orientation,
            source_size=size.swapped() if orientation.swaps_axes else size,
            upright_size=size,
        ),
        truncated=truncated,
    )


# --- selection -----------------------------------------------------------------------


def test_no_candidates_selects_nothing() -> None:
    assert select_primary((), FRAME) is None


def test_single_candidate_is_selected_regardless_of_strategy() -> None:
    only = face(0.0, 0.0, 5.0, 5.0)

    for strategy in SelectionStrategy:
        assert select_primary((only,), FRAME, strategy) is only


def test_largest_picks_the_biggest_box() -> None:
    small = face(0.0, 0.0, 20.0, 20.0)
    large = face(120.0, 120.0, 190.0, 190.0)

    assert select_primary((small, large), FRAME, SelectionStrategy.LARGEST) is large


def test_most_central_picks_the_centred_box() -> None:
    corner = face(0.0, 0.0, 60.0, 60.0)
    centred = face(85.0, 85.0, 115.0, 115.0)

    assert select_primary((corner, centred), FRAME, SelectionStrategy.MOST_CENTRAL) is centred


def test_most_confident_ignores_size() -> None:
    big = face(0.0, 0.0, 100.0, 100.0, score=0.5)
    sure = face(150.0, 150.0, 160.0, 160.0, score=0.99)

    assert select_primary((big, sure), FRAME, SelectionStrategy.MOST_CONFIDENT) is sure


def test_balanced_prefers_a_central_face_over_a_marginally_larger_edge_face() -> None:
    edge = face(0.0, 0.0, 62.0, 62.0)
    centred = face(70.0, 70.0, 130.0, 130.0)

    assert select_primary((edge, centred), FRAME, SelectionStrategy.BALANCED) is centred


def test_ties_break_on_confidence_for_determinism() -> None:
    """Reproducible benchmark runs depend on a stable answer for identical input."""
    lower = face(0.0, 0.0, 50.0, 50.0, score=0.7)
    higher = face(100.0, 100.0, 150.0, 150.0, score=0.95)

    assert select_primary((lower, higher), FRAME, SelectionStrategy.LARGEST) is higher


# --- stage ---------------------------------------------------------------------------


def test_no_face_is_a_status_not_an_exception() -> None:
    stage = DetectionStage(StubDetector(()))

    result = stage.run(solid_image(200, 200))

    assert result.status is DetectionStatus.NO_FACE
    assert not result.ok
    assert result.primary is None
    assert result.face_count == 0


def test_successful_run_reports_primary_and_provenance() -> None:
    stage = DetectionStage(StubDetector((face(80.0, 80.0, 130.0, 130.0),), name="stub-model"))

    result = stage.run(solid_image(200, 200))

    assert result.ok
    assert result.status is DetectionStatus.OK
    assert result.detector == "stub-model"
    assert result.duration_ms >= 0.0
    assert result.metadata["selection"] == "largest"
    assert result.image_size == FRAME


def test_multiple_faces_is_diagnosed_not_failed() -> None:
    stage = DetectionStage(
        StubDetector((face(20.0, 20.0, 60.0, 60.0), face(120.0, 120.0, 180.0, 180.0)))
    )

    result = stage.run(solid_image(200, 200))

    assert result.ok
    assert result.has(Diagnostic.MULTIPLE_FACES)
    assert result.face_count == 2


def test_low_confidence_is_flagged() -> None:
    stage = DetectionStage(StubDetector((face(80.0, 80.0, 130.0, 130.0, score=0.65),)))

    result = stage.run(solid_image(200, 200))

    assert result.has(Diagnostic.LOW_CONFIDENCE)


def test_small_face_is_flagged() -> None:
    stage = DetectionStage(StubDetector((face(90.0, 90.0, 98.0, 98.0),)))

    result = stage.run(solid_image(200, 200))

    assert result.has(Diagnostic.SMALL_FACE)


def test_face_touching_the_border_is_flagged() -> None:
    stage = DetectionStage(StubDetector((face(0.0, 40.0, 90.0, 150.0),)))

    result = stage.run(solid_image(200, 200))

    assert result.has(Diagnostic.FACE_TOUCHES_BORDER)


def test_centred_face_is_not_flagged_for_borders_or_size() -> None:
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0),)))

    result = stage.run(solid_image(200, 200))

    assert not result.has(Diagnostic.FACE_TOUCHES_BORDER)
    assert not result.has(Diagnostic.SMALL_FACE)
    assert not result.has(Diagnostic.LOW_CONFIDENCE)


def test_strong_roll_is_flagged() -> None:
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0, roll=25.0),)))

    result = stage.run(solid_image(200, 200))

    assert result.has(Diagnostic.STRONG_ROLL)


def test_mild_roll_is_not_flagged() -> None:
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0, roll=3.0),)))

    result = stage.run(solid_image(200, 200))

    assert not result.has(Diagnostic.STRONG_ROLL)


def test_orientation_and_truncation_provenance_reaches_the_result() -> None:
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0),)))
    image = loaded(solid_image(200, 200), orientation=ExifOrientation.RIGHT_TOP, truncated=True)

    result = stage.run(image)

    assert result.has(Diagnostic.ORIENTATION_CORRECTED)
    assert result.has(Diagnostic.TRUNCATED_IMAGE_DATA)


def test_raw_array_input_carries_no_load_provenance() -> None:
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0),)))

    result = stage.run(solid_image(200, 200))

    assert not result.has(Diagnostic.ORIENTATION_CORRECTED)
    assert not result.has(Diagnostic.TRUNCATED_IMAGE_DATA)


def test_stage_config_thresholds_are_honored() -> None:
    stage = DetectionStage(
        StubDetector((face(60.0, 50.0, 140.0, 150.0, score=0.85),)),
        StageConfig(low_confidence_below=0.9, selection=SelectionStrategy.BALANCED),
    )

    result = stage.run(solid_image(200, 200))

    assert result.has(Diagnostic.LOW_CONFIDENCE)
    assert result.metadata["selection"] == "balanced"


def test_stage_loads_from_a_path(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.fromarray(solid_image(200, 200), mode="RGB").save(path)
    stage = DetectionStage(StubDetector((face(60.0, 50.0, 140.0, 150.0),)))

    result = stage.run(path)

    assert result.ok
    assert result.image_size == FRAME


# --- detector construction -----------------------------------------------------------


def test_unknown_adapter_names_the_alternatives() -> None:
    with pytest.raises(ModelError, match="no detector adapter for 'nope'"):
        build_detector("nope")


def test_detector_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match=r"score_threshold must be in \[0, 1\]"):
        DetectorConfig(score_threshold=1.5)
    with pytest.raises(ValueError, match="max_faces must be at least 1"):
        DetectorConfig(max_faces=0)


# --- real weights, skipped when absent -----------------------------------------------


@pytest.fixture
def cached_default_detector():
    settings = load_settings()
    if not is_cached(DEFAULT_DETECTOR, settings):
        pytest.skip(f"{DEFAULT_DETECTOR} is not cached; run the CLI once to fetch it")
    return build_detector(DEFAULT_DETECTOR, settings=settings, allow_download=False)


def test_real_detector_validates_its_contract(cached_default_detector) -> None:
    """Construction validates the declared contract against the real signature."""
    assert cached_default_detector.name == DEFAULT_DETECTOR
    assert cached_default_detector.contract.input_name == "input"


def test_real_detector_finds_nothing_in_a_blank_frame(cached_default_detector) -> None:
    stage = DetectionStage(cached_default_detector)

    result = stage.run(solid_image(320, 320))

    assert result.status is DetectionStatus.NO_FACE
    assert result.duration_ms > 0.0
