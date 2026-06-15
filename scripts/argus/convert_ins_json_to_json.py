#!/usr/bin/env python
"""Convert Instagram post folders into unified ARGUS JSON.

Usage:

    python scripts/argus/convert_ins_json_to_json.py \
      --input-dir path/to/instagram/all \
      --output-dir inputs/argus/json/ins
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argus.path_utils import IMAGE_MEDIA_EXTENSIONS

SCHEMA_VERSION = "argus-user-json-v1"
PLATFORM = "instagram"
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


def convert_user_dir(user_dir: Path) -> dict[str, Any]:
    posts: list[dict[str, Any]] = []
    user_meta_candidates: list[dict[str, Any]] = []

    for post_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
        data_path = post_dir / "data.json"
        if not data_path.is_file():
            continue
        data = json.loads(data_path.read_text(encoding="utf-8"))
        post = _post_to_json(post_dir, data)
        posts.append(post)

        user_meta = _extract_user_metadata(data)
        if user_meta:
            user_meta_candidates.append(user_meta)

    posts.sort(key=_post_sort_key)
    user_metadata = _merge_user_metadata(user_meta_candidates)
    username = str(user_metadata.get("username") or user_dir.name)

    user: dict[str, Any] = {
        "user_id": user_dir.name,
        "username": username,
        "platform": PLATFORM,
        "metadata": {
            **user_metadata,
            "source_dir": str(user_dir),
        },
    }
    if user_metadata.get("full_name"):
        user["display_name"] = user_metadata["full_name"]
    if user_metadata.get("profile_pic_url"):
        user["avatar"] = {
            "type": "image",
            "path": user_metadata["profile_pic_url"],
            "source": PLATFORM,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": PLATFORM,
        "user": user,
        "posts": posts,
    }


def _post_to_json(post_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    info = data.get("info") or {}
    meta = data.get("meta") or {}
    basic = info.get("basic") or {}
    content = info.get("content") or {}
    stats = info.get("stats") or {}

    post_id = str(
        basic.get("id")
        or meta.get("pk")
        or meta.get("id")
        or post_dir.name
    )
    created_at = (
        basic.get("created_at")
        or meta.get("taken_at")
        or meta.get("created_at")
    )
    caption = (
        content.get("caption")
        or meta.get("caption_text")
        or ""
    )
    location = content.get("location")
    if location is None:
        location = meta.get("location")

    hashtags = content.get("hashtags")
    if hashtags is None:
        hashtags = _extract_hashtags_from_meta(meta)
    hashtags = _normalize_hashtags(hashtags)
    tagged_users = content.get("tagged_users")
    if tagged_users is None:
        tagged_users = _extract_tagged_users_from_meta(meta)
    tagged_users = _normalize_tagged_users(tagged_users)

    post: dict[str, Any] = {
        "post_id": post_id,
        "timestamp": created_at,
        "created_at": created_at,
        "caption": caption,
        "text": caption,
        "hashtags": hashtags,
        "tagged_users": tagged_users,
        "media_files": _local_media_files(post_dir),
        "metadata": {
            "post_dir": str(post_dir),
            "data_json": str(post_dir / "data.json"),
            "code": basic.get("code") or meta.get("code"),
            "type": basic.get("type"),
            "product_type": meta.get("product_type"),
            "media_type": meta.get("media_type"),
            "location": location,
            "hashtags": hashtags or [],
            "tagged_users": tagged_users or [],
            "stats": stats,
            "info": info,
            "meta": meta,
        },
    }
    if hashtags:
        post["tags"] = {"hashtags": hashtags}
    if location is not None:
        post["location"] = location
    return post


def _local_media_files(post_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(p for p in post_dir.iterdir() if p.is_file()):
        if path.name == "data.json" or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_MEDIA_EXTENSIONS:
            media_type = "image"
        elif suffix in VIDEO_EXTENSIONS:
            media_type = "video"
        else:
            continue
        out.append({
            "type": media_type,
            "path": str(path),
            "source": PLATFORM,
        })
    return out


def _extract_user_metadata(data: dict[str, Any]) -> dict[str, Any]:
    info = data.get("info") or {}
    meta = data.get("meta") or {}
    basic = info.get("basic") or {}
    author = basic.get("author") or {}
    meta_user = meta.get("user") or {}
    out: dict[str, Any] = {}
    if author.get("username") or meta_user.get("username"):
        out["username"] = author.get("username") or meta_user.get("username")
    if author.get("id") or meta_user.get("pk"):
        out["instagram_user_id"] = str(author.get("id") or meta_user.get("pk"))
    for key in ("full_name", "profile_pic_url", "profile_pic_url_hd", "is_private"):
        if key in meta_user:
            out[key] = meta_user[key]
    return out


def _merge_user_metadata(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in candidates:
        for key, value in item.items():
            if value not in (None, "") and key not in merged:
                merged[key] = value
    return merged


def _extract_hashtags_from_meta(meta: dict[str, Any]) -> list[str]:
    caption = str(meta.get("caption_text") or "")
    return [part[1:] for part in caption.split() if part.startswith("#") and len(part) > 1]


def _normalize_hashtags(raw: Any) -> list[str]:
    out: list[str] = []
    if not raw:
        return out
    for item in raw:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = str(item.get("name") or item.get("hashtag") or item.get("tag") or "").strip()
        else:
            value = str(item).strip()
        value = value.lstrip("#").strip()
        if value and value not in out:
            out.append(value)
    return out


def _extract_tagged_users_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in meta.get("usertags") or []:
        if not isinstance(item, dict):
            continue
        user = item.get("user") or {}
        if not isinstance(user, dict):
            continue
        out.append({
            "username": user.get("username"),
            "user_id": str(user.get("pk")) if user.get("pk") is not None else None,
            "full_name": user.get("full_name"),
            "position": {
                "x": item.get("x"),
                "y": item.get("y"),
            },
        })
    return out


def _normalize_tagged_users(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not raw:
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "user" in item and isinstance(item["user"], dict):
            user = item["user"]
            normalized = {
                "username": user.get("username"),
                "user_id": str(user.get("pk") or user.get("id")) if (user.get("pk") or user.get("id")) is not None else None,
                "full_name": user.get("full_name"),
                "position": {
                    "x": item.get("x"),
                    "y": item.get("y"),
                },
            }
        else:
            normalized = {
                "username": item.get("username"),
                "user_id": str(item.get("user_id") or item.get("id")) if (item.get("user_id") or item.get("id")) is not None else None,
                "position": item.get("position"),
            }
        if normalized.get("username") or normalized.get("user_id"):
            out.append(normalized)
    return out


def _post_sort_key(post: dict[str, Any]) -> tuple[int, str]:
    ts = post.get("timestamp")
    if isinstance(ts, str) and ts:
        try:
            return (0, datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat())
        except ValueError:
            return (1, ts)
    return (2, str(post.get("post_id") or ""))


def iter_user_dirs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Instagram user folders to unified ARGUS JSON.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Instagram user folders.",
    )
    parser.add_argument("--output-dir", default="inputs/argus/json/ins")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_dirs = iter_user_dirs(input_dir)
    if not user_dirs:
        raise SystemExit(f"No user directories found at {input_dir}")

    converted = 0
    total_posts = 0
    for user_dir in user_dirs:
        payload = convert_user_dir(user_dir)
        out_path = output_dir / f"{user_dir.name}.json"
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        n_posts = len(payload.get("posts") or [])
        converted += 1
        total_posts += n_posts
        print(f"[ins-json] {user_dir} -> {out_path} posts={n_posts}")

    print(f"[ins-json] done users={converted} posts={total_posts}")


if __name__ == "__main__":
    main()
