#!/usr/bin/env python
"""Convert synthetic generated users into unified ARGUS JSON.

Usage:

    python scripts/argus/convert_synthetic_json_to_json.py \
      --post-dir path/to/Synthetic_new/outputs/post \
      --image-dir path/to/Synthetic_new/final_img \
      --output-dir inputs/argus/json/synthetic

This script intentionally reads only:
  - the post script directory passed by --post-dir
  - the final image directory passed by --image-dir

It does not read Synthetic_new/outputs/profile, because that directory contains
ground-truth private attributes and must not leak into ARGUS inputs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argus.path_utils import IMAGE_MEDIA_EXTENSIONS


SCHEMA_VERSION = "argus-user-json-v1"
PLATFORM = "synthetic"
DEFAULT_OUTPUT_DIR = Path("inputs/argus/json/synthetic")


def convert_post_file(post_json_path: Path, image_root: Path) -> dict[str, Any]:
    data = json.loads(post_json_path.read_text(encoding="utf-8"))
    user_id = str(data.get("user_id") or _user_id_from_post_filename(post_json_path))
    account_info = data.get("account_info") or {}
    user_image_dir = image_root / user_id

    user: dict[str, Any] = {
        "user_id": user_id,
        "platform": PLATFORM,
        "metadata": {
            "account_info": account_info,
            "source_post_json": str(post_json_path),
            "image_dir": str(user_image_dir),
        },
    }
    nickname = account_info.get("nickname")
    if nickname:
        user["username"] = str(nickname)
        user["display_name"] = str(nickname)
    if account_info.get("ip_location"):
        user["ip_location"] = account_info["ip_location"]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": PLATFORM,
        "user": user,
        "posts": [
            _post_to_json(post, user_image_dir)
            for post in data.get("posts") or []
            if isinstance(post, dict)
        ],
    }


def _post_to_json(post: dict[str, Any], user_image_dir: Path) -> dict[str, Any]:
    post_id = str(post.get("post_id") or "")
    title = str(post.get("title") or "").strip()
    caption = str(post.get("caption") or "").strip()
    tags = _normalize_tags(post.get("tags"))
    image_ids = [str(item).strip() for item in (post.get("images") or []) if str(item).strip()]
    media_files, missing_images = _resolve_post_images(user_image_dir, image_ids)

    out: dict[str, Any] = {
        "post_id": post_id,
        "title": title,
        "caption": caption,
        "text": caption,
        "timestamp": post.get("post_time"),
        "created_at": post.get("post_time"),
        "tags": {"topics": tags} if tags else {},
        "topics": tags,
        "media_files": media_files,
        "metadata": {
            "post_time": post.get("post_time"),
            "tags": tags,
            "image_ids": image_ids,
            "missing_images": missing_images,
        },
    }
    return out


def _resolve_post_images(
    user_image_dir: Path,
    image_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    media_files: list[dict[str, Any]] = []
    missing: list[str] = []
    for image_id in image_ids:
        path = _find_image_file(user_image_dir, image_id)
        if path is None:
            missing.append(image_id)
            continue
        media_files.append({
            "type": "image",
            "path": str(path),
            "source": PLATFORM,
        })
    return media_files, missing


def _find_image_file(user_image_dir: Path, image_id: str) -> Path | None:
    raw = Path(image_id)
    candidates: list[Path] = []
    if raw.suffix.lower() in IMAGE_MEDIA_EXTENSIONS:
        candidates.append(user_image_dir / raw.name)
    else:
        for suffix in sorted(IMAGE_MEDIA_EXTENSIONS):
            candidates.append(user_image_dir / f"{image_id}{suffix}")
            candidates.append(user_image_dir / f"{image_id}{suffix.upper()}")
        candidates.extend(sorted(user_image_dir.glob(f"{image_id}.*")))

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_MEDIA_EXTENSIONS:
            return candidate
    return None


def _normalize_tags(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw_items = [raw]
    else:
        raw_items = list(raw)

    tags: list[str] = []
    for item in raw_items:
        tag = str(item).strip().lstrip("#")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _user_id_from_post_filename(path: Path) -> str:
    match = re.match(r"^(user_\d+)_post\.json$", path.name)
    if match:
        return match.group(1)
    return path.stem.replace("_post", "")


def convert_dir(post_dir: Path, image_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    post_files = sorted(post_dir.glob("user_*_post.json"))
    if not post_files:
        raise SystemExit(f"No user_*_post.json files found in {post_dir}")

    written = 0
    for post_json_path in post_files:
        converted = convert_post_file(post_json_path, image_dir)
        user_id = converted["user"]["user_id"]
        out_path = output_dir / f"{user_id}.json"
        out_path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        missing_count = sum(
            len((post.get("metadata") or {}).get("missing_images") or [])
            for post in converted.get("posts") or []
        )
        print(
            f"[synthetic-json] wrote {out_path} "
            f"posts={len(converted.get('posts') or [])} missing_images={missing_count}"
        )
        written += 1
    print(f"[synthetic-json] done users={written} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert synthetic users to ARGUS JSON inputs.")
    parser.add_argument("--post-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    convert_dir(
        Path(args.post_dir).expanduser(),
        Path(args.image_dir).expanduser(),
        Path(args.output_dir).expanduser(),
    )


if __name__ == "__main__":
    main()
