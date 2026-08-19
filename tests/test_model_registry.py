"""The pinned model catalog."""

from __future__ import annotations

import pytest

from portraitkit.errors import ModelError
from portraitkit.models.registry import (
    DEFAULT_DETECTOR,
    DEFAULT_MATTER,
    MODELS,
    ModelSpec,
    get_model,
    model_names,
)


def test_registry_is_not_empty() -> None:
    assert model_names()


def test_lookup_returns_the_matching_spec() -> None:
    spec = get_model(DEFAULT_DETECTOR)

    assert spec.name == DEFAULT_DETECTOR
    assert spec.filename.endswith(".onnx")


def test_lookup_matter_returns_matching_spec() -> None:
    spec = get_model(DEFAULT_MATTER)

    assert spec.name == DEFAULT_MATTER
    assert spec.filename.endswith(".onnx")


def test_unknown_model_names_the_alternatives() -> None:
    with pytest.raises(ModelError, match="unknown model 'nope'"):
        get_model("nope")


def test_registry_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        MODELS["injected"] = get_model(DEFAULT_DETECTOR)  # type: ignore[index]


@pytest.mark.parametrize("name", model_names())
def test_every_entry_is_pinned_and_verifiable(name: str) -> None:
    """A floating reference or a missing digest would silently break reproducibility."""
    spec = MODELS[name]

    assert spec.url.startswith("https://")
    assert len(spec.sha256) == 64
    assert spec.sha256 == spec.sha256.lower()
    assert spec.size_bytes > 0
    assert spec.license
    assert spec.license_url.startswith("https://")
    assert spec.upstream


@pytest.mark.parametrize("name", model_names())
def test_registry_key_matches_spec_name(name: str) -> None:
    assert MODELS[name].name == name


def test_filenames_are_unique() -> None:
    filenames = [spec.filename for spec in MODELS.values()]

    assert len(filenames) == len(set(filenames))


def test_default_detector_is_registered() -> None:
    assert DEFAULT_DETECTOR in MODELS


def test_default_matter_is_registered() -> None:
    assert DEFAULT_MATTER in MODELS


def test_default_detector_permits_commercial_use() -> None:
    """PortraitKit is MIT, so its default path must not carry a research-only weight.

    Several strong open detectors are released for non-commercial research only. Such a
    model may be offered, but promoting one to the default would make the library's
    license promise misleading for the integrators it targets.
    """
    assert get_model(DEFAULT_DETECTOR).permits_commercial_use


def test_default_matter_permits_commercial_use() -> None:
    """The default matting model must also permit commercial use."""
    assert get_model(DEFAULT_MATTER).permits_commercial_use


def test_research_only_entries_explain_the_restriction() -> None:
    for spec in MODELS.values():
        if not spec.permits_commercial_use:
            assert spec.notes, f"{spec.name} restricts use but explains nothing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "not-a-digest"),
        ("sha256", "A" * 64),
        ("url", "http://insecure.example/model.onnx"),
    ],
)
def test_malformed_specs_are_rejected(field: str, value: str) -> None:
    base = {
        "name": "probe",
        "filename": "probe.onnx",
        "url": "https://example.invalid/probe.onnx",
        "sha256": "0" * 64,
        "size_bytes": 10,
        "license": "MIT",
        "license_url": "https://example.invalid/LICENSE",
        "permits_commercial_use": True,
        "upstream": "test",
    }

    with pytest.raises(ValueError, match="probe"):
        ModelSpec(**{**base, field: value})


def test_non_positive_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="size_bytes must be positive"):
        ModelSpec(
            name="probe",
            filename="probe.onnx",
            url="https://example.invalid/probe.onnx",
            sha256="0" * 64,
            size_bytes=0,
            license="MIT",
            license_url="https://example.invalid/LICENSE",
            permits_commercial_use=True,
            upstream="test",
        )
