# PortraitKit

[![CI](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml/badge.svg)](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Make any photo a compliant, professional portrait — and measure how well it was done.

PortraitKit is an open portrait-processing pipeline: face detection, orientation handling, ICAO-style geometry cropping (Doc 9303), and optional background removal. Every stage ships with its evaluation twin in the same milestone, and **PortraitBench** — the project's evaluation harness — grades the open model zoo under realistic degradations.

**Status:** Stage 1 (detection and orientation) is complete with its evaluation module. Stages 2–4 are not built yet. There are no benchmark results to report; see [Roadmap](#roadmap).

## Who it's for

- **Everyday users** who want a passport, visa, or profile photo that meets real geometry and quality requirements without a studio.
- **Developers and integrators** who need a composable, measured portrait pipeline with reproducible quality evidence instead of vendor self-reporting.

## Pipeline

```
photo in
  1. Face detection + landmarks + orientation   [built]
  2. ICAO-style geometry crop                   [planned - Doc 9303 presets]
  3. Background removal / replacement           [planned - open matting adapters]
portrait out - with compliance and quality scores at every stage
```

## Quick start

```bash
git clone https://github.com/momtazularefin/portraitkit.git
cd portraitkit
uv sync --dev
```

Fetch the default detector (227 KiB, CPU-friendly):

```bash
uv run portraitkit fetch yunet-2023mar
```

Detect:

```bash
uv run portraitkit detect portrait.jpg
```

```
portrait.jpg: 1 face(s), primary [73, 47, 197, 214] score 0.954 (10.1 ms)
    roll -2.5 deg
    diagnostics: truncated_image_data
```

Add `--json` for machine-readable output, `--output results.json` to write a file, and `--model scrfd-10g-bnkps` to switch detectors. `--offline` refuses to touch the network.

### As a library

```python
from portraitkit.detection import DetectionStage, build_detector
from portraitkit.imaging.io import load_image

stage = DetectionStage(build_detector())
result = stage.run(load_image("portrait.jpg"))

if result.ok:
    print(result.primary.box, result.primary.landmarks.roll_degrees)
print([str(d) for d in result.diagnostics])
```

A stage never raises for an ordinary outcome. A photo with no face returns `status=no_face`, not an exception.

## Models

Model weights carry their own licenses, independent of PortraitKit's MIT license. The registry records that for every entry, and `portraitkit models` prints it:

| Model | Size | License | Commercial use | Role |
|---|---|---|---|---|
| `yunet-2023mar` | 227 KiB | MIT | Yes | Default |
| `scrfd-10g-bnkps` | 16.1 MiB | Non-commercial research only | **No** | Opt-in |

The default is MIT-licensed on purpose. Several strong open face detectors are released for non-commercial research only; promoting one to the default would make this repository's MIT promise misleading exactly where an integrator relies on it. SCRFD stays available and clearly flagged.

Every entry is pinned to an immutable upstream revision and a SHA-256 digest, verified before use. Weights are cached under `PORTRAITKIT_MODEL_DIR` (default `./models`) and are never committed.

## Evaluation

Stage 1 ships with its evaluation module, not a promise of one. Ground truth is a small JSON manifest that references images by relative path, so annotations travel without publishing a single photograph:

```json
{
  "schema_version": 1,
  "name": "example-set",
  "images": [
    {
      "path": "portraits/a.jpg",
      "faces": [
        {
          "box": [60, 40, 160, 180],
          "primary": true,
          "landmarks": [[85, 80], [135, 80], [110, 110], [90, 140], [130, 140]]
        }
      ]
    }
  ]
}
```

```bash
uv run portraitkit evaluate annotations.json --output report.json
```

Reported: precision, recall, F1, mean IoU, landmark error normalized by interocular distance, and **primary-selection accuracy**. That last one matters — a pipeline must choose a subject on every multi-face photo, and until the choice is scored it is folklore. It reports "not annotated" rather than a perfect score when no image declares a primary.

Reports are deterministic: manifest order preserved, floats rounded, and images that fail to load recorded as errors rather than dropped, so a partial run cannot read as a clean sweep.

### What has actually been measured

Honest scope: these are engineering-validation results, not benchmark results.

- **Orientation invariance.** Across 20 portraits rendered under all eight EXIF orientation values (160 variants), the stage recovered the upright face box with a minimum IoU of 0.952 and a median of 0.994, with zero failures. Residual variation is JPEG re-encoding, not coordinate error.
- **CPU latency**, 260×300 inputs, single thread, ONNX Runtime CPU provider: YuNet ≈ 10 ms median, SCRFD ≈ 112 ms median.
- **Decoder correctness.** Both detector decoders are pinned by hand-built tensors with exact expected geometry, independent of any model file, and cross-validate against each other on real photographs.

No public-dataset leaderboard exists yet. Scoring against labelled public data is milestone M4.

## Design notes

**Preprocessing is declared, not assumed.** Every model adapter carries a `PreprocessContract` — input name, size, colour order, tensor layout, mean, scale, resize mode — validated against the signature ONNX Runtime actually reports, at load time. This exists because the predecessor work to this project shipped one model behind two wrappers that normalized incompatibly, with nothing in either able to say which was right.

**One inference boundary.** Everything runs through ONNX Runtime. A benchmark that measured one model through PyTorch and another through ONNX Runtime would be reporting runtime differences alongside model quality with no way to separate them.

**Shared postprocessing.** Score filtering, non-maximum suppression, coordinate inversion, and clipping live in the adapter base class. Adapters differ only in their contract and their decode, so a comparison reflects the models rather than their wrappers.

## PortraitBench independence

PortraitBench lives in this repository and grades third-party open models reached through adapters. PortraitKit contributes no model of its own to the rankings. The one PortraitKit-authored component the harness will score is the ICAO cropper, and it is scored by [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project), the external reference implementation of ISO/IEC 29794-5 — not by this project. Every result will be reproducible from its versioned config.

## Privacy

No portrait, crop, embedding, or per-identity record enters this repository, its history, its documentation, or any published artifact. Tests use synthetic images generated at run time; there are no checked-in photographs.

## Roadmap

- **M1** — Face detection, landmarks, orientation + evaluation module — **done**
- **M2** — ICAO geometry cropper with OFIQ-scored compliance evidence
- **M3** — Background removal: open matting adapters + SAD/MSE/Grad/Conn metrics
- **M4** — PortraitBench assembly: degradation suite, reproducible public leaderboard
- **M5** — ONNX artifacts, .NET consumption sample, results report
- **M6** — GUI application

## Configuration

All paths come from the environment; nothing is hard-coded. See [.env.example](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `PORTRAITKIT_MODEL_DIR` | `./models` | Model weight cache |
| `PORTRAITKIT_DATA_DIR` | `./data` | Evaluation datasets |
| `PORTRAITKIT_OUTPUT_DIR` | `./output` | Pipeline and benchmark output |
| `PORTRAITKIT_ALLOW_DOWNLOAD` | `1` | Set to `0` to forbid network fetches |

## License

[MIT](LICENSE). Model weights are licensed separately by their upstream authors; see [Models](#models).
