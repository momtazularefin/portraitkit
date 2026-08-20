# PortraitKit

[![CI](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml/badge.svg)](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)
[![.NET: 8.0+](https://img.shields.io/badge/.NET-8.0%2B-purple.svg)](samples/dotnet/PortraitKit.Sample)

Make any photo a compliant, professional portrait — and measure how well it was done.

PortraitKit is an open portrait-processing pipeline: face detection, orientation handling, ISO/IEC 39794-5 (ICAO Doc 9303) geometry cropping, and background removal. Every stage ships with its evaluation twin in the same milestone, and **PortraitBench** — the project's evaluation harness — grades the open model zoo under realistic image degradations.

**Status:** Milestones M1 through M5 are complete. Stages 1, 2, and 3 are fully operational with evaluation twins, the PortraitBench degradation harness is assembled, ONNX models are standardized, and a native C#/.NET 8+ consumption sample is provided.

---

## Who it's for

- **Everyday users** who want a passport, visa, or profile photo that meets international geometry and quality requirements without a photo studio.
- **Developers and ML engineers** who need a composable, measured portrait pipeline with reproducible quality evidence instead of vendor self-reporting.

---

## Architecture & Pipeline

```
                       [ Input Photo (Arbitrary EXIF / Rotation) ]
                                           │
                                           ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Stage 1: Detection & Orientation                                                │
  │ • EXIF & 90° rotation recovery • Multi-face score filtering • Primary selection │
  │ Models: YuNet (MIT, default), SCRFD-10G (Non-commercial)                        │
  └──────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Stage 2: ISO/IEC 39794-5 Geometry Cropping & Quality Assessment                 │
  │ • Table D.8 target aspect & face positioning • Synthetic derotation             │
  │ Scorer: Official EU eu-LISA OFIQ 1.0.3 reference runner                         │
  └──────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Stage 3: Background Matting & Replacement                                       │
  │ • Alpha matte estimation • Solid color fill (passport) • RGBA transparency     │
  │ Models: BiRefNet, MODNet, RMBG-1.4, U²-Net, U²-Netp, IS-Net                     │
  └──────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
                       [ Verified Compliant Output Portrait ]

  ───────────────────────────────────────────────────────────────────────────────────
  Cross-Cutting Tooling:
  • PortraitBench: Parameterized degradation harness (JPEG, Blur, Noise, Low-Light)
  • .NET Satellite: Cross-runtime ONNX Runtime inference sample in C# / .NET 8+
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/momtazularefin/portraitkit.git
cd portraitkit
uv sync --dev
```

### 1. Fetch Models

```bash
# Fetch default detector (YuNet, 227 KiB, MIT)
uv run portraitkit fetch yunet-2023mar

# Fetch default matting model (MODNet, 24.7 MiB, Apache-2.0)
uv run portraitkit fetch modnet-photographic
```

### 2. Detect & Crop

```bash
# Detect primary face and landmarks
uv run portraitkit detect portrait.jpg --json

# Crop to ICAO 35x45mm passport preset
uv run portraitkit crop portrait.jpg --preset icao-portrait-35x45 --output passport.png
```

### 3. Remove Background (Matting)

```bash
# Replace background with solid white (passport compliant)
uv run portraitkit matte portrait.jpg --bg-color white --output passport_white.png

# Generate transparent RGBA profile cutout
uv run portraitkit matte portrait.jpg --bg-color transparent --output profile_cutout.png
```

### 4. Open the Inspector (GUI)

```bash
uv run portraitkit gui
```

Opens a local inspector at `http://127.0.0.1:8000`. Drop a portrait, or click one of the
bundled synthetic samples, and it shows every stage timing, the detected landmarks, the
crop rectangle, and the full geometry conformance table with the clause behind each check.

The overlay encodes evidence strength visually: **solid** marks are measured from detected
landmarks, **dashed** marks are inferred crown and chin positions. That matches the
`measured` / `estimated` badge on every row of the conformance table, so what you see and
what the table claims cannot drift apart.

It runs on the Python standard library. PortraitKit's default detector is 227 KiB and its
dependency list is four packages; shipping a web framework or a desktop toolkit to display
that would have cost more than the thing it displays.

### 5. Run PortraitBench Evaluations

```bash
# Inspect available parameterized degradations
uv run portraitbench degradations

# Execute benchmark run across models and corruptions
uv run portraitbench run --config configs/portraitbench-v1.json --output results/bench-run.json

# Render Markdown leaderboard
uv run portraitbench report results/bench-run.json
```

---

## Model Zoo

Model weights carry their own licenses, independent of PortraitKit's MIT license. Every entry is pinned to an immutable upstream SHA-256 digest:

| Model | Task | Size | Input Size | License | Commercial Use | Role |
|---|---|---|---|---|---|---|
| `yunet-2023mar` | Detection | 227 KiB | Dynamic | MIT | **Yes** | Default Detector |
| `scrfd-10g-bnkps` | Detection | 16.1 MiB | Dynamic | Non-commercial | No | Opt-in Research |
| `modnet-photographic` | Matting | 24.7 MiB | $512 \times 512$ | Apache-2.0 | **Yes** | Default Matter |
| `rmbg-1.4` | Matting | 176 MiB | $1024 \times 1024$ | BRIA Non-Comm | No | High-Detail Research |
| `birefnet-general` | Matting | 98.4 MiB | $1024 \times 1024$ | Apache-2.0 | **Yes** | SOTA Boundary |
| `u2net-human-seg` | Matting | 176 MiB | $320 \times 320$ | Apache-2.0 | **Yes** | Robust Human Mask |
| `u2netp` | Matting | 4.4 MiB | $320 \times 320$ | Apache-2.0 | **Yes** | Mobile / Edge |
| `isnet-general-use` | Matting | 176 MiB | $1024 \times 1024$ | Apache-2.0 | **Yes** | High-Res Silhouette |

---

## What Has Actually Been Measured

PortraitBench is a benchmark project, so this section states precisely what has been run
and what has not. Every number below is reproducible from a versioned artifact or a
documented command.

### Measured

| Result | Value | How to reproduce |
|---|---|---|
| Detection latency, CPU, 260x300 inputs | YuNet ~10 ms median, SCRFD ~112 ms median | `portraitkit detect <images> --model ...` |
| Orientation invariance, all 8 EXIF values over 20 portraits (160 variants) | min IoU 0.952, median 0.994, zero failures | see `tests/test_orientation.py` and the M1 evidence |
| Decoder cross-validation | YuNet and SCRFD agree on eye landmarks within a few pixels on identical inputs | `tests/test_detection_decode.py` |
| ICAO crop vs. external OFIQ 1.0.3 referee, 10 synthetic portraits | head size +63.5, inter-eye +16.0, roll +0.5, margins -52.0 / -97.5 | [`results/m2b-ofiq-synthetic-v1.json`](results/m2b-ofiq-synthetic-v1.json) |

The OFIQ run is the project's substantive finding and is analysed below.

### Not yet measured

**No model-quality leaderboard has been produced.** Ranking detectors by F1 under
degradation, or matting models by SAD, MSE, gradient, and connectivity error, requires
ground-truth annotations and alpha mattes that this project has not yet acquired. The
harness, the degradation suite, the adapters, and all four metrics are implemented and
tested, so the missing input is data rather than code.

Publishing a ranking without that data would make this project the thing it was built to
replace: a benchmark reporting numbers nobody can check. The tables will appear here when
a licensed public dataset backs them, and not before.


## External OFIQ Quality Scoring & Findings

PortraitKit integrates **OFIQ 1.0.3** (ISO/IEC 29794-5 reference implementation) via an isolated subprocess runner:

```bash
uv run portraitkit ofiq fetch
uv run portraitkit ofiq score portrait.jpg --offline
```

### The Margin Arithmetic Paradox
Scoring ICAO Table D.8 crops against OFIQ revealed a structural arithmetic tension:
- OFIQ's native margin metrics $\frac{Y}{T}$ (top) and $\frac{B-Y}{T}$ (bottom) sum identically to $\frac{B}{T} = \frac{1}{\text{Head Size}}$.
- Satisfying both margins ($\ge 3.2$) forces Head Size $\le 0.3125$, where the head-size scalar drops to **12/100**.
- Table D.8-conformant passport crops (crown-to-chin 60–90%) maintain $T/B \ge 0.35$, mathematically precluding high OFIQ margin scores.
- PortraitBench preserves and documents this tension rather than treating quality and geometry as interchangeable.

---

## Cross-Runtime .NET Satellite

PortraitKit includes a C# sample in [`samples/dotnet/PortraitKit.Sample`](samples/dotnet/PortraitKit.Sample) demonstrating native ONNX model consumption using `Microsoft.ML.OnnxRuntime`:

```bash
cd samples/dotnet/PortraitKit.Sample
dotnet build
dotnet run
```

---

## Design Principles

1. **Declared Preprocess Contract**: Input geometry, normalization factors, and resize modes (`LETTERBOX_PAD` vs `BILINEAR_STRETCH`) are declared in code and validated against ONNX Runtime reported tensor shapes at load time.
2. **Unified ONNX Inference Boundary**: Eliminates framework runtime discrepancies during benchmark scoring.
3. **Complementary Metrics**: SAD, MSE, Gradient, and Connectivity errors quantify distinct failure modes (mass error, localized holes, edge blur, detached floaters).

---

## PortraitBench Independence Statement

PortraitBench lives in this repository and grades third-party open models reached through uniform adapter interfaces. PortraitKit authors no proprietary contestant model. The authored ICAO cropper is graded by the external reference implementation [OFIQ 1.0.3](https://github.com/BSI-OFIQ/OFIQ-Project/tree/v1.0.3). All benchmark runs are reproducible from versioned JSON configs.

---

## Roadmap

- [x] **M1** — Face detection, 5-point landmarks, EXIF orientation recovery & evaluation module.
- [x] **M2** — ISO/IEC 39794-5 geometry cropper with external OFIQ reference scoring.
- [x] **M3** — Background removal: open matting adapters & 4-metric evaluation twin.
- [x] **M4** — PortraitBench assembly: degradation suite, configuration runner, and CLI.
- [x] **M5** — ONNX standardized artifacts, C#/.NET 8+ sample, and technical results report.
- [x] **M6** — Local inspector GUI exposing per-stage timings, overlays, and the conformance table.

---

## License

[MIT](LICENSE). Model weights are licensed separately by upstream authors.
