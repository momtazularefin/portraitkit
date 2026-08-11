# PortraitKit

<!-- Badges will be added after GitHub repo setup -->
<!-- ![CI](https://github.com/momtazularefin/portraitkit/actions/workflows/ci.yml/badge.svg) -->
<!-- ![License](https://img.shields.io/github/license/momtazularefin/portraitkit) -->

Make any photo a compliant, professional portrait — and measure how well it was done.

PortraitKit is an open portrait-processing pipeline: face detection, orientation handling, ICAO-style geometry cropping (Doc 9303), and optional background removal. Every stage ships with its evaluation twin, and **PortraitBench** — the project's evaluation harness — grades the open model zoo on public datasets under realistic degradations.

**Status:** 🏗️ Under active development (pre-M1).

## Who it's for

- **Everyday users** who want a passport, visa, or profile photo (LinkedIn, CV) that meets real geometry and quality requirements without a studio.
- **Developers and integrators** who need a composable, measured portrait pipeline with reproducible quality evidence instead of vendor self-reporting.

## Pipeline

```
photo in
  1. Face detection + landmarks + orientation   (adapted proven detectors)
  2. ICAO-style geometry crop                   (Doc 9303 presets: passport, visa, profile)
  3. Background removal / replacement           (optional; open matting model adapters)
portrait out — with compliance and quality scores at every stage
```

Compliance scoring builds on [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project), the open reference implementation of ISO/IEC 29794-5 face image quality.

## PortraitBench

The evaluation harness behind the pipeline:

- Standard matting metrics (SAD, MSE, Grad, Conn) on public datasets (P3M-10k, AM-2k, AIM-500).
- A degradation suite — compression, blur, low light, noise, low resolution, non-standard backgrounds — because production photos are not studio photos.
- Reproducible leaderboard runs over open background-removal and detection models via a uniform adapter interface.
- ONNX Runtime as the common inference boundary, with latency/throughput reporting.

## Stack

| Layer | Tools |
|-------|-------|
| Inference | ONNX Runtime, PyTorch (model adapters) |
| Imaging | OpenCV, Pillow |
| Compliance | OFIQ (ISO/IEC 29794-5), ICAO Doc 9303 geometry |
| Evaluation | PortraitBench harness, public matting datasets |
| Tooling | Python 3.11+, uv, ruff, pytest |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/momtazularefin/portraitkit.git
cd portraitkit
uv sync --dev

# Run tests
uv run pytest
```

Pipeline and benchmark CLIs land milestone by milestone; see the roadmap below.

## Roadmap

- **M1** — Face detection, landmarks, and orientation stage + its evaluation module
- **M2** — ICAO geometry cropper with OFIQ-scored compliance evidence
- **M3** — Background removal stage: open model adapters + matting metrics
- **M4** — PortraitBench assembly: degradation suite, reproducible leaderboard
- **M5** — ONNX artifacts, .NET consumption sample, results report

## Results

> Benchmark results will be published as milestones complete.

## License

[MIT](LICENSE) — see the [LICENSE](LICENSE) file.
