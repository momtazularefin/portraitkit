"""Non-maximum suppression over candidate boxes.

Implemented here rather than pulled from a detector library so that every adapter is
suppressed identically. If one PortraitBench entrant used a library's NMS and another
used its own, part of the measured difference between them would be the suppression, not
the model.
"""

from __future__ import annotations

import numpy as np

__all__ = ["non_max_suppression"]


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
    top_k: int | None = None,
) -> np.ndarray:
    """Return indices of boxes to keep, highest score first.

    Args:
        boxes: ``(N, 4)`` array of ``x1, y1, x2, y2``.
        scores: ``(N,)`` array of confidences.
        iou_threshold: Overlap above which a lower-scoring box is suppressed.
        top_k: Consider only this many highest-scoring candidates. Bounds the cost on
            dense feature maps, where most candidates are noise.

    Returns:
        A 1-D integer array of surviving indices into ``boxes``.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        msg = f"expected an (N, 4) box array, got shape {boxes.shape}"
        raise ValueError(msg)
    if scores.shape != (boxes.shape[0],):
        msg = f"expected {boxes.shape[0]} scores, got shape {scores.shape}"
        raise ValueError(msg)
    if not 0.0 <= iou_threshold <= 1.0:
        msg = f"iou_threshold must be in [0, 1], got {iou_threshold}"
        raise ValueError(msg)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = (boxes[:, index].astype(np.float64) for index in range(4))
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores.astype(np.float64))[::-1]
    if top_k is not None:
        order = order[:top_k]

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

        union = areas[current] + areas[rest] - intersection
        # Two zero-area boxes have no meaningful overlap; treat them as disjoint rather
        # than dividing by zero.
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = rest[overlap <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)
