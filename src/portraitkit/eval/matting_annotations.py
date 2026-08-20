"""Ground-truth annotation manifests for matting evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from portraitkit.errors import AnnotationError

__all__ = ["MattingAnnotationSet", "MattingSample", "load_matting_annotations"]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MattingSample:
    """One paired ground-truth matting sample."""

    image_path: Path
    alpha_path: Path
    trimap_path: Path | None
    relative_image: str
    relative_alpha: str
    relative_trimap: str | None = None


@dataclass(frozen=True, slots=True)
class MattingAnnotationSet:
    """A collection of ground-truth matting samples."""

    name: str
    samples: tuple[MattingSample, ...]
    root: Path


def load_matting_annotations(
    manifest_path: str | Path, *, root: str | Path | None = None
) -> MattingAnnotationSet:
    """Load a matting annotation manifest.

    Args:
        manifest_path: Path to the JSON manifest.
        root: Directory image/mask paths are relative to. Defaults to the manifest's parent.

    Returns:
        A loaded :class:`MattingAnnotationSet`.
    """
    path = Path(manifest_path)
    if not path.exists():
        msg = f"manifest {str(path)!r} does not exist"
        raise AnnotationError(msg)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        msg = f"failed to parse manifest {str(path)!r}: {error}"
        raise AnnotationError(msg) from error

    if not isinstance(data, dict):
        msg = f"manifest {str(path)!r} must be a JSON object"
        raise AnnotationError(msg)

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION}"
        raise AnnotationError(msg)

    name = data.get("name", path.stem)
    base_root = Path(root) if root is not None else path.parent

    raw_samples = data.get("samples") or data.get("images")
    if not isinstance(raw_samples, list):
        msg = f"manifest {str(path)!r} must contain a 'samples' list"
        raise AnnotationError(msg)

    parsed_samples: list[MattingSample] = []
    for item in raw_samples:
        if not isinstance(item, dict):
            continue
        img_rel = item.get("image") or item.get("path")
        alpha_rel = item.get("alpha") or item.get("mask") or item.get("matte")
        trimap_rel = item.get("trimap")

        if not img_rel or not alpha_rel:
            msg = "each sample must specify 'image' and 'alpha' paths"
            raise AnnotationError(msg)

        parsed_samples.append(
            MattingSample(
                image_path=base_root / img_rel,
                alpha_path=base_root / alpha_rel,
                trimap_path=base_root / trimap_rel if trimap_rel else None,
                relative_image=str(img_rel),
                relative_alpha=str(alpha_rel),
                relative_trimap=str(trimap_rel) if trimap_rel else None,
            )
        )

    return MattingAnnotationSet(
        name=name,
        samples=tuple(parsed_samples),
        root=base_root,
    )
