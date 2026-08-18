"""The ONNX Runtime inference boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from portraitkit.errors import ModelError
from portraitkit.models.session import (
    CPU_PROVIDER,
    available_providers,
    create_session,
    describe_session,
    select_providers,
)
from tests.conftest import write_tiny_onnx


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    return write_tiny_onnx(tmp_path / "tiny.onnx")


def test_cpu_provider_is_always_available() -> None:
    assert CPU_PROVIDER in available_providers()


def test_default_selection_is_cpu_only() -> None:
    assert select_providers() == (CPU_PROVIDER,)


def test_gpu_preference_falls_back_to_cpu_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reporting a GPU run that never happened would corrupt latency comparisons."""
    monkeypatch.setattr("portraitkit.models.session.available_providers", lambda: (CPU_PROVIDER,))

    assert select_providers(prefer_gpu=True) == (CPU_PROVIDER,)


def test_gpu_preference_puts_the_accelerator_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "portraitkit.models.session.available_providers",
        lambda: ("CUDAExecutionProvider", CPU_PROVIDER),
    )

    assert select_providers(prefer_gpu=True) == ("CUDAExecutionProvider", CPU_PROVIDER)


def test_session_loads_and_runs(tiny_model: Path) -> None:
    session = create_session(tiny_model)

    payload = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    (result,) = session.run(None, {"x": payload})

    assert np.array_equal(result, payload)


def test_session_runs_on_the_cpu_provider(tiny_model: Path) -> None:
    session = create_session(tiny_model)

    assert CPU_PROVIDER in session.get_providers()


def test_describe_reports_the_signature(tiny_model: Path) -> None:
    info = describe_session(create_session(tiny_model))

    assert [spec.name for spec in info.inputs] == ["x"]
    assert [spec.name for spec in info.outputs] == ["y"]
    assert info.inputs[0].dtype == "tensor(float)"
    assert info.inputs[0].shape == (1, 3)
    assert not info.inputs[0].is_dynamic
    assert CPU_PROVIDER in info.providers


def test_describe_flags_dynamic_dimensions(tmp_path: Path) -> None:
    model = write_tiny_onnx(tmp_path / "dynamic.onnx", dynamic_batch=True)

    info = describe_session(create_session(model))

    assert info.inputs[0].is_dynamic


def test_input_lookup_by_name(tiny_model: Path) -> None:
    info = describe_session(create_session(tiny_model))

    assert info.input_named("x").dtype == "tensor(float)"


def test_input_lookup_names_the_alternatives(tiny_model: Path) -> None:
    info = describe_session(create_session(tiny_model))

    with pytest.raises(ModelError, match="available inputs are: x"):
        info.input_named("images")


def test_missing_model_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="no ONNX model at"):
        create_session(tmp_path / "absent.onnx")


def test_unreadable_model_raises_model_error(tmp_path: Path) -> None:
    path = tmp_path / "garbage.onnx"
    path.write_bytes(b"not a protobuf")

    with pytest.raises(ModelError, match="could not load ONNX model"):
        create_session(path)


def test_thread_count_must_be_positive(tiny_model: Path) -> None:
    with pytest.raises(ValueError, match="intra_op_threads must be at least 1"):
        create_session(tiny_model, intra_op_threads=0)


def test_pinned_thread_count_is_accepted(tiny_model: Path) -> None:
    session = create_session(tiny_model, intra_op_threads=1)

    payload = np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32)
    (result,) = session.run(None, {"x": payload})

    assert np.array_equal(result, payload)


def test_repeated_runs_are_identical(tiny_model: Path) -> None:
    session = create_session(tiny_model, deterministic=True)
    payload = np.asarray([[0.25, -1.5, 7.0]], dtype=np.float32)

    first = session.run(None, {"x": payload})[0]
    second = session.run(None, {"x": payload})[0]

    assert np.array_equal(first, second)
