"""The crop evaluator publishes aggregate external scores, never source rows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from portraitkit.crop.ofiq import OfiqProvenance
from portraitkit.eval.crop import evaluate_crop_quality
from tests.conftest import solid_image


def test_crop_quality_evaluation_is_aggregate_and_reproducible(tmp_path: Path) -> None:
    sources = (tmp_path / "private-name-a.jpg", tmp_path / "private-name-b.jpg")
    for path in sources:
        Image.fromarray(solid_image(80, 100)).save(path)

    detector = SimpleNamespace(name="yunet-2023mar")

    class FakeDetectionStage:
        def __init__(self) -> None:
            self.detector = detector

        def run(self, _image):
            return SimpleNamespace(primary=object())

    class FakeCropStage:
        def __init__(self) -> None:
            self.preset = SimpleNamespace(name="icao-portrait-35x45")
            self.calls = 0

        def run(self, _image, _detection):
            self.calls += 1
            return SimpleNamespace(
                ok=True,
                image=solid_image(35, 45),
                conforms=True,
                padded=self.calls == 1,
                status="ok",
            )

    provenance = OfiqProvenance(
        version="1.0.3",
        source_revision="a" * 40,
        package_sha256="b" * 64,
        executable_sha256="c" * 64,
        config_sha256="d" * 64,
        models_sha256="e" * 64,
        platform="win64",
    )

    class FakeScorer:
        def __init__(self) -> None:
            self.provenance = provenance

        def score(self, directory: Path):
            results = []
            for path in sorted(directory.glob("*.png")):
                scalar = 80.0 if path.name.endswith("after.png") else 50.0
                measurement = SimpleNamespace(
                    name="HeadSize", native_score=1.0, scalar_score=scalar, assessed=True
                )
                results.append(SimpleNamespace(image=path, measurements=(measurement,)))
            return tuple(results)

    report = evaluate_crop_quality(
        FakeDetectionStage(),
        FakeCropStage(),
        FakeScorer(),
        sources,
        dataset="synthetic-public-set",
    )
    target = tmp_path / "results" / "report.json"
    report.write(target)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["dataset"]["name"] == "synthetic-public-set"
    assert payload["input_count"] == 2
    assert payload["scored_pairs"] == 2
    assert payload["conforming_crops"] == 2
    assert payload["padded_crops"] == 1
    assert payload["measures"]["HeadSize"]["median_delta"] == 30.0
    assert payload["measures"]["HeadSize"]["improved"] == 2
    assert payload["detector"]["artifact_sha256"].startswith("8f2383e4")
    assert len(payload["input_set_sha256"]) == 64
    serialized = target.read_text(encoding="utf-8")
    assert "private-name" not in serialized
