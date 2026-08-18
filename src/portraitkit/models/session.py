"""ONNX Runtime session construction: the project's single inference boundary.

Every model in PortraitKit and every entrant in PortraitBench executes through this
module. That uniformity is not incidental tidiness: a benchmark that measured one model
through PyTorch and another through ONNX Runtime would be reporting runtime differences
alongside model quality and could not honestly separate them.

CPU execution is the required path. A GPU provider is used only when explicitly asked
for and actually present, so results stay reproducible on ordinary hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import onnxruntime as ort

from portraitkit.errors import ModelError

__all__ = [
    "CPU_PROVIDER",
    "SessionInfo",
    "TensorSpec",
    "available_providers",
    "create_session",
    "describe_session",
    "select_providers",
]

CPU_PROVIDER: Final = "CPUExecutionProvider"
"""The provider every supported platform has, and the one benchmark runs report."""

_GPU_PROVIDER_PREFERENCE: Final = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One model input or output as ONNX Runtime reports it."""

    name: str
    shape: tuple[int | str | None, ...]
    dtype: str

    @property
    def is_dynamic(self) -> bool:
        """Whether any dimension is symbolic rather than a fixed size."""
        return any(not isinstance(dimension, int) for dimension in self.shape)


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """A structural summary of a loaded model."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    providers: tuple[str, ...]

    def input_named(self, name: str) -> TensorSpec:
        """Return the input called ``name``.

        Raises:
            ModelError: If the model has no such input.
        """
        for spec in self.inputs:
            if spec.name == name:
                return spec
        available = ", ".join(spec.name for spec in self.inputs)
        msg = f"model has no input named {name!r}; available inputs are: {available}"
        raise ModelError(msg)


def available_providers() -> tuple[str, ...]:
    """Return the execution providers this ONNX Runtime build offers."""
    return tuple(ort.get_available_providers())


def select_providers(*, prefer_gpu: bool = False) -> tuple[str, ...]:
    """Choose an execution provider list.

    CPU is always included as the final fallback. When ``prefer_gpu`` is set, the first
    available accelerator is placed ahead of it; if none is present the result is simply
    the CPU provider, because silently reporting a GPU run that never happened would
    corrupt any latency comparison built on it.
    """
    if not prefer_gpu:
        return (CPU_PROVIDER,)
    installed = set(available_providers())
    for candidate in _GPU_PROVIDER_PREFERENCE:
        if candidate in installed:
            return (candidate, CPU_PROVIDER)
    return (CPU_PROVIDER,)


def create_session(
    model_path: str | Path,
    *,
    providers: tuple[str, ...] | None = None,
    intra_op_threads: int | None = None,
    deterministic: bool = True,
) -> ort.InferenceSession:
    """Load ``model_path`` into an ONNX Runtime session.

    Args:
        model_path: Path to a local ``.onnx`` file, normally from
            :func:`portraitkit.models.store.resolve_model`.
        providers: Execution providers in priority order. Defaults to CPU only.
        intra_op_threads: Thread count for intra-operator parallelism. Leave unset for
            the runtime default; pin it when comparing latency across models.
        deterministic: Disable memory-pattern reuse so repeated runs on identical input
            produce identical output. Benchmark reproducibility outranks the small
            throughput cost.

    Returns:
        A ready inference session.

    Raises:
        ModelError: If the file is missing or cannot be loaded as an ONNX model.
    """
    path = Path(model_path)
    if not path.is_file():
        msg = f"no ONNX model at {path}"
        raise ModelError(msg)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is not None:
        if intra_op_threads < 1:
            msg = f"intra_op_threads must be at least 1, got {intra_op_threads}"
            raise ValueError(msg)
        options.intra_op_num_threads = intra_op_threads
    if deterministic:
        options.enable_mem_pattern = False

    try:
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=list(providers or (CPU_PROVIDER,)),
        )
    # onnxruntime raises several unrelated exception types for a bad artifact; all of
    # them mean the same thing to a caller.
    except Exception as error:
        msg = f"could not load ONNX model {path}: {error}"
        raise ModelError(msg) from error


def describe_session(session: ort.InferenceSession) -> SessionInfo:
    """Summarize a session's input and output signature.

    Adapters use this to assert their declared tensor contract against the artifact they
    actually loaded, which is how a preprocessing mismatch becomes a loud failure instead
    of a silently wrong result.
    """
    return SessionInfo(
        inputs=tuple(
            TensorSpec(name=node.name, shape=tuple(node.shape), dtype=node.type)
            for node in session.get_inputs()
        ),
        outputs=tuple(
            TensorSpec(name=node.name, shape=tuple(node.shape), dtype=node.type)
            for node in session.get_outputs()
        ),
        providers=tuple(session.get_providers()),
    )
