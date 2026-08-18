"""Ground-truth annotation sets for detection evaluation.

An annotation set is a small JSON manifest that names images by relative path and lists
the faces each one contains. Images stay outside the repository; the manifest is the only
thing that travels, which is what lets an evaluation be reproducible without ever
publishing a portrait.

One face per image may be marked ``primary``. That flag is what makes primary-subject
selection measurable rather than assumed: the pipeline has to choose a subject on every
multi-face photo, and until the choice is scored it is folklore.

Manifest shape::

    {
      "schema_version": 1,
      "name": "example-set",
      "images": [
        {
          "path": "portraits/a.jpg",
          "faces": [
            {"box": [60, 40, 160, 180], "primary": true,
             "landmarks": [[85, 80], [135, 80], [110, 110], [90, 140], [130, 140]]}
          ]
        }
      ]
    }

Boxes are ``[x1, y1, x2, y2]`` in upright-image pixels. ``landmarks`` is optional and,
when present, holds five ``[x, y]`` pairs in detector point order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portraitkit.errors import AnnotationError
from portraitkit.types import BoundingBox, FaceLandmarks5

__all__ = [
    "AnnotatedFace",
    "AnnotatedImage",
    "AnnotationSet",
    "load_annotations",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AnnotatedFace:
    """One ground-truth face."""

    box: BoundingBox
    landmarks: FaceLandmarks5 | None = None
    primary: bool = False


@dataclass(frozen=True, slots=True)
class AnnotatedImage:
    """One annotated image and the faces it contains."""

    relative_path: str
    path: Path
    faces: tuple[AnnotatedFace, ...]

    @property
    def primary(self) -> AnnotatedFace | None:
        """The face marked primary, if the annotation declares one."""
        for face in self.faces:
            if face.primary:
                return face
        return None


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """A named collection of annotated images."""

    name: str
    root: Path
    images: tuple[AnnotatedImage, ...]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def face_count(self) -> int:
        """Total annotated faces across every image."""
        return sum(len(image.faces) for image in self.images)

    @property
    def images_with_primary(self) -> int:
        """How many images declare a primary subject."""
        return sum(1 for image in self.images if image.primary is not None)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationError(message)


def _parse_box(raw: Any, where: str) -> BoundingBox:
    _require(
        isinstance(raw, list | tuple) and len(raw) == 4,
        f"{where}: box must be a list of four numbers, got {raw!r}",
    )
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{where}: box values must be numeric: {error}") from error
    try:
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    except ValueError as error:
        raise AnnotationError(f"{where}: {error}") from error


def _parse_landmarks(raw: Any, where: str) -> FaceLandmarks5 | None:
    if raw is None:
        return None
    _require(
        isinstance(raw, list | tuple) and len(raw) == 5,
        f"{where}: landmarks must be five [x, y] pairs, got {raw!r}",
    )
    try:
        points = np.asarray([[float(x), float(y)] for x, y in raw], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{where}: landmark values must be numeric: {error}") from error
    return FaceLandmarks5.from_array(points)


def _parse_face(raw: Any, where: str) -> AnnotatedFace:
    _require(isinstance(raw, dict), f"{where}: face must be an object, got {raw!r}")
    _require("box" in raw, f"{where}: face is missing its box")
    return AnnotatedFace(
        box=_parse_box(raw["box"], where),
        landmarks=_parse_landmarks(raw.get("landmarks"), where),
        primary=bool(raw.get("primary", False)),
    )


def _parse_image(raw: Any, index: int, root: Path) -> AnnotatedImage:
    where = f"image {index}"
    _require(isinstance(raw, dict), f"{where}: entry must be an object, got {raw!r}")
    relative = raw.get("path")
    _require(
        isinstance(relative, str) and relative.strip() != "",
        f"{where}: path must be a non-empty string",
    )
    assert isinstance(relative, str)
    where = f"image {index} ({relative})"

    faces_raw = raw.get("faces", [])
    _require(isinstance(faces_raw, list), f"{where}: faces must be a list")
    faces = tuple(_parse_face(face, where) for face in faces_raw)

    primary_count = sum(1 for face in faces if face.primary)
    _require(
        primary_count <= 1,
        f"{where}: at most one face may be marked primary, found {primary_count}",
    )
    return AnnotatedImage(relative_path=relative, path=root / relative, faces=faces)


def load_annotations(manifest_path: str | Path, root: str | Path | None = None) -> AnnotationSet:
    """Load an annotation manifest.

    Args:
        manifest_path: Path to the JSON manifest.
        root: Directory image paths are relative to. Defaults to the manifest's own
            directory, so a manifest sitting beside its images just works.

    Returns:
        The parsed annotation set. Image files are not opened here; a manifest can be
        validated without the images present.

    Raises:
        AnnotationError: If the manifest is missing, is not valid JSON, declares an
            unsupported schema version, or contains a malformed entry.
    """
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AnnotationError(f"cannot read annotation manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise AnnotationError(f"annotation manifest {path} is not valid JSON: {error}") from error

    _require(isinstance(payload, dict), f"{path}: manifest must be a JSON object")
    version = payload.get("schema_version")
    _require(
        version == SCHEMA_VERSION,
        f"{path}: unsupported schema_version {version!r}, expected {SCHEMA_VERSION}",
    )

    images_raw = payload.get("images")
    _require(isinstance(images_raw, list), f"{path}: images must be a list")
    assert isinstance(images_raw, list)

    resolved_root = Path(root) if root is not None else path.parent
    images = tuple(
        _parse_image(entry, index, resolved_root) for index, entry in enumerate(images_raw)
    )

    name = payload.get("name") or path.stem
    _require(isinstance(name, str), f"{path}: name must be a string")
    return AnnotationSet(name=str(name), root=resolved_root, images=images)
