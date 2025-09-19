"""COCO-format utilities shared across core datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


def load_coco(path: str | Path) -> Mapping[str, Any]:
    """Load a COCO JSON file from ``path`` (relative paths are resolved).

    Parameters
    ----------
    path:
        Path to the JSON annotation file.
    """

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        return json.load(handle)


def index_images(coco: Mapping[str, Any]) -> Dict[int, MutableMapping[str, Any]]:
    """Build a fast lookup of image records by their integer id."""

    images = coco.get("images", []) or []
    indexed: Dict[int, MutableMapping[str, Any]] = {}
    for record in images:
        try:
            image_id = int(record["id"])
        except Exception:
            continue
        indexed[image_id] = dict(record)
    return indexed


def resolve_category_id(coco: Mapping[str, Any], name_or_id: str | int) -> int | None:
    """Resolve a category id from either a name or explicit id."""

    categories: Sequence[Mapping[str, Any]] = coco.get("categories", []) or []
    if isinstance(name_or_id, str):
        target = name_or_id.lower()
        for cat in categories:
            if str(cat.get("name", "")).lower() == target:
                return int(cat.get("id"))
        return None

    # Already an id – confirm it exists.
    try:
        cat_id = int(name_or_id)
    except Exception:
        return None
    for cat in categories:
        if int(cat.get("id", -1)) == cat_id:
            return cat_id
    return None


def collect_annotations_by_image(
    coco: Mapping[str, Any],
    *,
    category_ids: Iterable[int] | int | None = None,
) -> Dict[int, list[Mapping[str, Any]]]:
    """Group annotations by ``image_id`` with optional category filtering."""

    anns = coco.get("annotations", []) or []
    if isinstance(category_ids, Iterable) and not isinstance(category_ids, (str, bytes)):
        allowed = {int(cat_id) for cat_id in category_ids}
    elif category_ids is None:
        allowed = None
    else:
        allowed = {int(category_ids)}

    grouped: Dict[int, list[Mapping[str, Any]]] = {}
    for ann in anns:
        try:
            image_id = int(ann["image_id"])
        except Exception:
            continue
        if allowed is not None and int(ann.get("category_id", -1)) not in allowed:
            continue
        grouped.setdefault(image_id, []).append(ann)
    return grouped


__all__ = [
    "collect_annotations_by_image",
    "index_images",
    "load_coco",
    "resolve_category_id",
]
