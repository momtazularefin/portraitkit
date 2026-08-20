"""Stage 3: background removal, and the metrics that grade it."""

from portraitkit.matting.base import (
    MattingAdapter,
    build_matter,
    register_matting_adapter,
)
from portraitkit.matting.birefnet import (
    BIREFNET_CONTRACT,
    BiRefNetAdapter,
    decode_birefnet,
)
from portraitkit.matting.isnet import (
    ISNET_CONTRACT,
    ISNetAdapter,
    decode_isnet,
)
from portraitkit.matting.metrics import (
    GRADIENT_SIGMA,
    MattingMetrics,
    connectivity_error,
    gradient_error,
    matting_metrics,
    mean_squared_error,
    sum_absolute_difference,
)
from portraitkit.matting.modnet import (
    MODNET_CONTRACT,
    MODNetAdapter,
    decode_modnet,
)
from portraitkit.matting.rmbg import (
    RMBG14_CONTRACT,
    RMBG14Adapter,
    decode_rmbg,
)
from portraitkit.matting.stage import (
    MattingResult,
    MattingStage,
    MattingStageConfig,
    composite_matte,
    parse_color,
)
from portraitkit.matting.u2net import (
    U2NET_CONTRACT,
    U2NetAdapter,
    U2NetPocketAdapter,
    decode_u2net,
)

__all__ = [
    "BIREFNET_CONTRACT",
    "GRADIENT_SIGMA",
    "ISNET_CONTRACT",
    "MODNET_CONTRACT",
    "RMBG14_CONTRACT",
    "U2NET_CONTRACT",
    "BiRefNetAdapter",
    "ISNetAdapter",
    "MODNetAdapter",
    "MattingAdapter",
    "MattingMetrics",
    "MattingResult",
    "MattingStage",
    "MattingStageConfig",
    "RMBG14Adapter",
    "U2NetAdapter",
    "U2NetPocketAdapter",
    "build_matter",
    "composite_matte",
    "connectivity_error",
    "decode_birefnet",
    "decode_isnet",
    "decode_modnet",
    "decode_rmbg",
    "decode_u2net",
    "gradient_error",
    "matting_metrics",
    "mean_squared_error",
    "parse_color",
    "register_matting_adapter",
    "sum_absolute_difference",
]
