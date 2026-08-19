"""Environment-based configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from portraitkit.config import load_settings
from portraitkit.errors import ConfigError


def test_defaults_are_relative_paths() -> None:
    settings = load_settings(env={})

    assert settings.model_dir == Path("models")
    assert settings.data_dir == Path("data")
    assert settings.output_dir == Path("output")
    assert settings.ofiq_dir == Path("models/ofiq")
    assert settings.allow_download is True
    assert not settings.model_dir.is_absolute()


def test_environment_overrides_are_honored() -> None:
    settings = load_settings(
        env={
            "PORTRAITKIT_MODEL_DIR": "/srv/weights",
            "PORTRAITKIT_DATA_DIR": "/srv/datasets",
            "PORTRAITKIT_OUTPUT_DIR": "/srv/out",
            "PORTRAITKIT_OFIQ_DIR": "/srv/ofiq",
            "PORTRAITKIT_ALLOW_DOWNLOAD": "false",
        }
    )

    assert settings.model_dir == Path("/srv/weights")
    assert settings.data_dir == Path("/srv/datasets")
    assert settings.output_dir == Path("/srv/out")
    assert settings.ofiq_dir == Path("/srv/ofiq")
    assert settings.allow_download is False


def test_blank_values_fall_back_to_defaults() -> None:
    settings = load_settings(env={"PORTRAITKIT_MODEL_DIR": "   ", "PORTRAITKIT_ALLOW_DOWNLOAD": ""})

    assert settings.model_dir == Path("models")
    assert settings.ofiq_dir == Path("models/ofiq")
    assert settings.allow_download is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_download_values(value: str) -> None:
    assert load_settings(env={"PORTRAITKIT_ALLOW_DOWNLOAD": value}).allow_download is True


@pytest.mark.parametrize("value", ["0", "false", "No", "off"])
def test_falsy_download_values(value: str) -> None:
    assert load_settings(env={"PORTRAITKIT_ALLOW_DOWNLOAD": value}).allow_download is False


def test_unparseable_boolean_raises() -> None:
    with pytest.raises(ConfigError, match="PORTRAITKIT_ALLOW_DOWNLOAD"):
        load_settings(env={"PORTRAITKIT_ALLOW_DOWNLOAD": "maybe"})


def test_ensure_directories_creates_them(tmp_path: Path) -> None:
    settings = load_settings(
        env={
            "PORTRAITKIT_MODEL_DIR": str(tmp_path / "m"),
            "PORTRAITKIT_DATA_DIR": str(tmp_path / "d"),
            "PORTRAITKIT_OUTPUT_DIR": str(tmp_path / "o" / "nested"),
        }
    )

    settings.ensure_directories()

    assert settings.model_dir.is_dir()
    assert settings.data_dir.is_dir()
    assert settings.output_dir.is_dir()
    assert settings.ofiq_dir.is_dir()
