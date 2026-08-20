"""The inspector's analysis service.

The service is deliberately HTTP-free, so everything here runs without opening a socket.
Cases needing real weights self-skip when the detector is not cached, matching the rest of
the suite.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from portraitkit.config import load_settings
from portraitkit.gui.service import AnalysisService, encode_png, render_overlay
from portraitkit.imaging.io import load_image
from portraitkit.models.registry import DEFAULT_DETECTOR
from portraitkit.models.store import is_cached
from portraitkit.types import DetectionResult, DetectionStatus, ImageSize
from tests.conftest import noise_image, solid_image


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def weights_required() -> None:
    if not is_cached(DEFAULT_DETECTOR, load_settings()):
        pytest.skip(f"{DEFAULT_DETECTOR} is not cached")


# --- options --------------------------------------------------------------------------


def test_options_are_derived_from_the_registries() -> None:
    """Hard-coding the lists would let the UI drift from what is actually installable."""
    options = AnalysisService().options().to_dict()

    names = {entry["name"] for entry in options["detectors"]}
    assert "yunet-2023mar" in names
    assert options["default_preset"] in {p["name"] for p in options["presets"]}


def test_options_carry_the_licence_restriction() -> None:
    """A reviewer choosing a detector should see the research-only limit in the picker."""
    detectors = {d["name"]: d for d in AnalysisService().options().to_dict()["detectors"]}

    assert detectors["yunet-2023mar"]["commercial"] is True
    assert detectors["scrfd-10g-bnkps"]["commercial"] is False


def test_options_are_json_serializable() -> None:
    json.dumps(AnalysisService().options().to_dict())


# --- rendering ------------------------------------------------------------------------


def test_encode_png_produces_a_data_url() -> None:
    url = encode_png(solid_image(8, 8))

    assert url.startswith("data:image/png;base64,")
    assert len(url) > 40


def test_overlay_survives_an_image_with_no_face() -> None:
    """A blank frame must render, not raise; the UI has to show the empty result."""
    loaded = load_image(png_bytes(solid_image(64, 64)))
    empty = DetectionResult(status=DetectionStatus.NO_FACE, image_size=ImageSize(64, 64))

    overlay = render_overlay(loaded, empty, None)

    assert overlay.shape == (64, 64, 3)


def test_overlay_downscales_a_large_source() -> None:
    """Full-resolution previews would push megabytes of base64 into every response."""
    loaded = load_image(png_bytes(noise_image(1600, 1600)))
    empty = DetectionResult(status=DetectionStatus.NO_FACE, image_size=ImageSize(1600, 1600))

    overlay = render_overlay(loaded, empty, None)

    assert max(overlay.shape[:2]) == 900


# --- analysis -------------------------------------------------------------------------


def test_unreadable_upload_returns_an_error_payload_not_an_exception() -> None:
    """The UI renders whatever it gets back; an exception would show a blank panel."""
    result = AnalysisService(allow_download=False).analyse(b"definitely not an image")

    assert result["ok"] is False
    assert "error" in result


def test_blank_frame_reports_no_face(weights_required: None) -> None:
    result = AnalysisService(allow_download=False).analyse(png_bytes(solid_image(320, 320)))

    assert result["detection"]["status"] == "no_face"
    assert result["crop"]["status"] == "no_face"
    assert result["overlay"].startswith("data:image/png")


def test_analysis_payload_is_json_serializable(weights_required: None) -> None:
    result = AnalysisService(allow_download=False).analyse(png_bytes(solid_image(320, 320)))

    json.dumps(result)


def test_detector_sessions_are_reused(weights_required: None) -> None:
    """Building an ONNX session costs far more than an inference; per-request
    construction would make the UI feel slow for reasons unrelated to the pipeline."""
    service = AnalysisService(allow_download=False)
    blob = png_bytes(solid_image(320, 320))

    service.analyse(blob)
    first = service._stage_for("yunet-2023mar")
    service.analyse(blob)

    assert service._stage_for("yunet-2023mar") is first


# --- samples --------------------------------------------------------------------------


def test_samples_are_empty_when_the_set_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PORTRAITKIT_DATA_DIR", str(tmp_path))

    assert AnalysisService().samples() == ()
    assert AnalysisService().sample_bytes(0) is None


def test_samples_are_discovered_and_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "public-samples" / "set"
    root.mkdir(parents=True)
    Image.fromarray(solid_image(32, 32), mode="RGB").save(root / "a.jpg")
    monkeypatch.setenv("PORTRAITKIT_DATA_DIR", str(tmp_path))

    service = AnalysisService()

    assert len(service.samples()) == 1
    assert service.sample_bytes(0)[:2] == b"\xff\xd8"  # JPEG start-of-image
    assert service.sample_bytes(5) is None


def test_negative_sample_index_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Python indexing would happily wrap a negative index onto a real file."""
    root = tmp_path / "public-samples"
    root.mkdir(parents=True)
    Image.fromarray(solid_image(32, 32), mode="RGB").save(root / "a.jpg")
    monkeypatch.setenv("PORTRAITKIT_DATA_DIR", str(tmp_path))

    assert AnalysisService().sample_bytes(-1) is None
