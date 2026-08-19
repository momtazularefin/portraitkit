"""Model resolution and the ONNX Runtime inference boundary."""

from portraitkit.models.contract import (
    ColorOrder,
    PreprocessContract,
    ResizeMode,
    TensorLayout,
)
from portraitkit.models.registry import DEFAULT_DETECTOR, MODELS, ModelSpec, get_model, model_names
from portraitkit.models.session import (
    CPU_PROVIDER,
    SessionInfo,
    TensorSpec,
    available_providers,
    create_session,
    describe_session,
    select_providers,
)
from portraitkit.models.store import cached_path, file_digest, is_cached, resolve_model

__all__ = [
    "CPU_PROVIDER",
    "DEFAULT_DETECTOR",
    "MODELS",
    "ColorOrder",
    "ModelSpec",
    "PreprocessContract",
    "ResizeMode",
    "SessionInfo",
    "TensorLayout",
    "TensorSpec",
    "available_providers",
    "cached_path",
    "create_session",
    "describe_session",
    "file_digest",
    "get_model",
    "is_cached",
    "model_names",
    "resolve_model",
    "select_providers",
]
