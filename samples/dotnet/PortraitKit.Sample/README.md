# PortraitKit .NET Consumption Sample

This sample demonstrates how external .NET services and C# applications can directly consume standardized ONNX models provided by **PortraitKit** using `Microsoft.ML.OnnxRuntime`.

## Highlights

- **Direct ONNX Interoperability**: Consumes the exact same ONNX models (`yunet-2023mar.onnx`, `modnet-photographic.onnx`, `birefnet-general.onnx`, etc.) used in the Python pipeline.
- **Contract-Compliant Tensor Execution**: Implements the `PreprocessContract` standard (NCHW tensor layout, float32 normalization).
- **Zero Python Dependency in Production**: Demonstrates cross-runtime deployment where Python is used for research/evaluation and .NET runs native inference.

## Quick Start

### 1. Build the sample

```bash
dotnet build
```

### 2. Run mock validation

```bash
dotnet run
```

### 3. Run inference against a real ONNX model

Download any supported model using the PortraitKit CLI:

```bash
uv run portraitkit fetch yunet-2023mar
```

Then run the .NET sample passing the model path:

```bash
dotnet run -- ../../../models/yunet-2023mar.onnx
```
