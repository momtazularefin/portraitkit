"""Smoke test — verify the package is importable."""

from portraitkit import __version__


def test_version():
    assert __version__ == "0.1.0"
