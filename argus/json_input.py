"""Unified JSON input loader for ARGUS experiments.

The loader normalizes XHS, Instagram, and synthetic exports into the post shape
expected by ARGUSPipeline:

    {"post_id": str, "caption": str, "media_files": [path, ...],
     "timestamp": str | None, "location_ip": str | None}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from argus.config import project_root
from argus.path_utils import is_image_media_ref, normalize_media_ref

SCHEMA_VERSION = "argus-user-json-v1"


def load_user_json(path: str | Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Load one unified ARGUS user JSON file.

    Relative media paths are resolved first from the JSON file directory, then
    from the project root/current working directory. Absolute paths are kept.
    Non-image media entries are ignored.
    """
    json_path = Path(path).expanduser()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    user = data.get("user") or {}

    user_id = str(
        user.get("user_id")
        or data.get("user_id")
        or user.get("username")
        or json_path.stem
    )
    metadata = _metadata_from_json(data, user, json_path)

    posts: list[dict[str, Any]] = []
    for idx, post in enumerate(data.get("posts") or [], start=1):
        post_id = str(post.get("post_id") or post.get("id") or f"post_{idx:03d}")
        caption = _caption_from_post(post)
        media_files = _resolve_media_files(
            post.get("media_files")
            or post.get("media")
            or post.get("images")
            or [],
            base_dir=json_path.parent,
        )
        post_metadata = dict(post.get("metadata") or {})
        for key in ("tags", "topics", "hashtags", "tagged_users"):
            if key in post:
                post_metadata[key] = post[key]
        posts.append({
            "post_id": post_id,
            "caption": caption,
            "media_files": media_files,
            "timestamp": post.get("timestamp") or post.get("created_at"),
            "location_ip": post.get("location_ip") or post.get("ip_location"),
            "metadata": post_metadata,
        })

    return user_id, metadata, posts


def iter_user_json_files(input_path: str | Path) -> list[Path]:
    """Return JSON inputs from one file or a directory."""
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.glob("*.json")
        if p.is_file() and p.name != "manifest.json"
    )


def _metadata_from_json(
    data: dict[str, Any],
    user: dict[str, Any],
    json_path: Path,
) -> dict[str, Any]:
    metadata = dict(user.get("metadata") or {})
    for key in ("source", "schema_version"):
        if key in data:
            metadata[key] = data[key]
    for key in ("username", "platform", "display_name"):
        if key in user:
            metadata[key] = user[key]
    for key in ("tags", "tag_list"):
        if key in user:
            metadata[key] = user[key]
    if "ip_location" in user:
        metadata["user_ip_location"] = user["ip_location"]
    metadata.setdefault("schema_version", SCHEMA_VERSION)
    metadata["input_json"] = str(json_path)
    return metadata


def _caption_from_post(post: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "caption", "text", "description"):
        value = post.get(key)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return "\n".join(parts)


def _resolve_media_files(raw_media: Any, *, base_dir: Path) -> list[str]:
    if isinstance(raw_media, (str, dict)):
        items = [raw_media]
    else:
        items = list(raw_media or [])

    resolved: list[str] = []
    for item in items:
        media_type, ref = _media_item_type_and_ref(item)
        if not ref:
            continue
        if media_type and media_type not in {"image", "photo"}:
            continue
        media_ref = _resolve_media_ref(ref, base_dir)
        if is_image_media_ref(media_ref):
            resolved.append(media_ref)
    return resolved


def _media_item_type_and_ref(item: Any) -> tuple[str | None, str]:
    if isinstance(item, str):
        return None, item
    if isinstance(item, dict):
        media_type = item.get("type") or item.get("media_type") or item.get("kind")
        ref = (
            item.get("path")
            or item.get("file")
            or item.get("url")
            or item.get("src")
            or ""
        )
        return str(media_type).lower() if media_type else None, str(ref)
    return None, ""


def _resolve_media_ref(ref: str, base_dir: Path) -> str:
    ref = normalize_media_ref(str(ref).strip())
    if ref.startswith(("http://", "https://", "data:")):
        return ref

    path = Path(ref).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        base_dir / path,
        project_root() / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((base_dir / path).resolve())
