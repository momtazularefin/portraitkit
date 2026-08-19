# PortraitKit

[![CI](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml/badge.svg)](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Make any photo a compliant, professional portrait — and measure how well it was done.

PortraitKit is an open portrait-processing pipeline: face detection, orientation handling, ICAO-style geometry cropping (Doc 9303), and optional background removal. Every stage ships with its evaluation twin in the same milestone, and **PortraitBench** — the project's evaluation harness — grades the open model zoo under realistic degradations.

**Status:** Stages 1 and 2 are complete with their evaluation twins. The crop stage is
measured by the pinned OFIQ 1.0.3 reference implementation; background removal and the
full public leaderboard remain future work. See [Roadmap](#roadmap).

## Who it's for

- **Everyday users** who want a passport, visa, or profile photo that meets real geometry and quality requirements without a studio.
- **Developers and integrators** who need a composable, measured portrait pipeline with reproducible quality evidence instead of vendor self-reporting.

## Pipeline

```
photo in
  1. Face detection + landmarks + orientation   [built]
  2. ISO/IEC 39794-5 geometry crop + OFIQ       [built]
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

### External OFIQ quality scoring

OFIQ is optional because its official reference package is about 997 MiB. Fetching it
installs the checksum-pinned eu-LISA package in the ignored local model cache; it does not
need Docker or a Visual Studio build:

```bash
uv run portraitkit ofiq fetch
uv run portraitkit ofiq score portrait.jpg --offline
```

PortraitKit invokes `OFIQSampleApp` as a subprocess, never through in-process native
bindings. JSON results carry the exact OFIQ version, tagged source revision, package,
executable, configuration, model-tree, and platform hashes. Normal tests and CI remain
offline and do not require OFIQ. On this machine the Win64 package reproduced all 784
scalar values across the 28 official conformance images exactly.

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
- **External crop quality.** A deterministic selection of 10 identity-free synthetic
  portraits was detected, cropped to `icao-portrait-35x45`, and scored before and after by
  OFIQ 1.0.3. All 10 pairs scored; nine crops passed every geometry check and one correctly
  reported that padding was needed. Median OFIQ scalar changes were: head size **+63.5**,
  inter-eye distance **+16.0**, roll **+0.5**, and unified quality **+0.5**. The complete
  aggregate is [versioned here](results/m2b-ofiq-synthetic-v1.json).

The crop run also found a negative result worth keeping visible: median OFIQ top- and
bottom-margin scores fell by 52.0 and 97.5. PortraitKit's preset targets the licensed
ISO/IEC 39794-5 Table D.8 geometry; OFIQ measures ISO/IEC 29794-5 face-image quality. One
does not certify the other, and this sample shows that satisfying the former can conflict
with preferences measured by the latter. The unified score improved only slightly and on
five of ten samples, so this is engineering evidence, not a claim that every input gets
better.

The sample selection is CC-BY-4.0 synthetic imagery pinned by URL, size, SHA-256, and
upstream Git revision in
[`configs/m2b-ofiq-synthetic.json`](configs/m2b-ofiq-synthetic.json). Source images and
crops stay in ignored local caches; only aggregate scores are versioned. Reproduce the run
with:

```bash
uv run portraitkit fetch yunet-2023mar
uv run portraitkit ofiq fetch
uv run portraitkit ofiq evaluate-crop \
  --manifest configs/m2b-ofiq-synthetic.json \
  --output results/m2b-ofiq-synthetic-v1.json
```

No public-dataset leaderboard exists yet. Scoring against labelled public data is milestone M4.

## Design notes

**Preprocessing is declared, not assumed.** Every model adapter carries a `PreprocessContract` — input name, size, colour order, tensor layout, mean, scale, resize mode — validated against the signature ONNX Runtime actually reports, at load time. This exists because the predecessor work to this project shipped one model behind two wrappers that normalized incompatibly, with nothing in either able to say which was right.

**One inference boundary.** Everything runs through ONNX Runtime. A benchmark that measured one model through PyTorch and another through ONNX Runtime would be reporting runtime differences alongside model quality with no way to separate them.

**Shared postprocessing.** Score filtering, non-maximum suppression, coordinate inversion, and clipping live in the adapter base class. Adapters differ only in their contract and their decode, so a comparison reflects the models rather than their wrappers.

## PortraitBench independence

PortraitBench lives in this repository and grades third-party open models reached through
adapters. PortraitKit contributes no model of its own to the rankings. Its authored
cropper is scored by [OFIQ 1.0.3](https://github.com/BSI-OFIQ/OFIQ-Project/tree/v1.0.3),
the last release upstream identifies as the ISO/IEC 29794-5 reference implementation —
not by PortraitKit. Version 1.0.3 is intentional: newer OFIQ releases are conformant
implementations but [are no longer the reference
implementation](https://github.com/BSI-OFIQ/OFIQ-Project/blob/v1.1.0/CHANGELOG.md).
Every published result carries scorer, model, input-set, and configuration provenance.

## Privacy

No portrait, crop, embedding, or per-identity record enters this repository, its history, its documentation, or any published artifact. Tests use synthetic images generated at run time; there are no checked-in photographs.

## Roadmap

- **M1** — Face detection, landmarks, orientation + evaluation module — **done**
- **M2** — ISO/IEC 39794-5 geometry cropper with external OFIQ evidence — **done**
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
| `PORTRAITKIT_OFIQ_DIR` | `./models/ofiq` | Optional OFIQ package cache |
| `PORTRAITKIT_ALLOW_DOWNLOAD` | `1` | Set to `0` to forbid network fetches |

## License

[MIT](LICENSE). Model weights are licensed separately by their upstream authors; see [Models](#models).
