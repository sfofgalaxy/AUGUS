#!/usr/bin/env python
"""Convert Xiaohongshu user_notes txt files into unified ARGUS JSON.

Usage:

    python scripts/argus/convert_xhs_txt_to_json.py \
      --input-dir path/to/xhs/user_notes \
      --output-dir inputs/argus/json/xhs
"""
from __future__ import annotations

import argparse
import ast
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
PLATFORM = "xiaohongshu"
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
}


def convert_file(txt_path: Path) -> dict[str, Any]:
    text = txt_path.read_text(encoding="utf-8")
    basic_text, posts_text = _split_sections(text)
    basic = _parse_basic_block(basic_text)
    posts = _parse_posts_block(posts_text)

    user_id = txt_path.stem
    tags = _coerce_tags(basic.get("tag_list"))
    user: dict[str, Any] = {
        "user_id": user_id,
        "platform": PLATFORM,
        "metadata": basic,
    }
    nickname = basic.get("nickname")
    if nickname:
        user["username"] = str(nickname)
        user["display_name"] = str(nickname)
    if tags is not None:
        user["tags"] = tags
    if basic.get("current_ip_location"):
        user["ip_location"] = basic["current_ip_location"]
    if basic.get("avatar"):
        user["avatar"] = _media_item(str(basic["avatar"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "source": PLATFORM,
        "user": user,
        "posts": [_post_to_json(post) for post in posts],
    }


def _split_sections(text: str) -> tuple[str, str]:
    parts = text.split("Posts (chronological):", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _parse_basic_block(text: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key is not None:
            fields[_normalize_key(current_key)] = "\n".join(current_lines).strip()
        current_key = None
        current_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.lower() == "basic:":
            continue
        match = re.match(r"^\s*-\s*([^:]+):\s*(.*)$", line)
        if match:
            flush()
            current_key = match.group(1).strip()
            current_lines = [_unwrap_ref(match.group(2).strip())]
        elif current_key is not None:
            current_lines.append(_unwrap_ref(stripped))
    flush()
    return {key: _parse_scalar(value) for key, value in fields.items()}


def _parse_posts_block(text: str) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_key: str | None = None
    media_mode = False

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.setdefault("media", [])
            posts.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if current is not None and current_key and current_key != "media":
                current[current_key] = f"{current.get(current_key, '')}\n".rstrip()
            continue

        post_match = re.match(r"^\s*-\s*post_id:\s*(.*)$", line)
        if post_match:
            flush()
            current = {"post_id": post_match.group(1).strip(), "media": []}
            current_key = "post_id"
            media_mode = False
            continue
        if current is None:
            continue

        if media_mode and _looks_like_media_item(stripped):
            current.setdefault("media", []).append(_unwrap_ref(stripped.lstrip("- ").strip()))
            continue

        field_match = re.match(r"^\s{2,}([^:]+):\s*(.*)$", line)
        if field_match:
            key = _normalize_key(field_match.group(1).strip())
            value = _unwrap_ref(field_match.group(2).strip())
            if key == "media":
                current.setdefault("media", [])
                current_key = "media"
                media_mode = True
            else:
                current[key] = value
                current_key = key
                media_mode = False
            continue

        if current_key and current_key != "media":
            previous = str(current.get(current_key, "")).rstrip()
            current[current_key] = f"{previous}\n{_unwrap_ref(stripped)}".strip()

    flush()
    return posts


def _post_to_json(post: dict[str, Any]) -> dict[str, Any]:
    fields = {k: _parse_scalar(v) for k, v in post.items() if k != "media"}
    post_id = str(fields.get("post_id") or "")
    topics = _extract_xhs_topics(
        "\n".join(
            str(fields.get(key) or "")
            for key in ("title", "text")
        )
    )
    media_files = [
        _media_item(str(ref))
        for ref in post.get("media", [])
        if str(ref).strip()
    ]

    out: dict[str, Any] = {
        "post_id": post_id,
        "media_files": media_files,
        "metadata": fields,
    }
    if topics:
        out["topics"] = topics
        out["tags"] = {"topics": topics}
        out["metadata"]["topics"] = topics
    if fields.get("create_time"):
        out["timestamp"] = fields["create_time"]
        out["created_at"] = fields["create_time"]
    if fields.get("last_update_time"):
        out["updated_at"] = fields["last_update_time"]
    if fields.get("title") is not None:
        out["title"] = fields["title"]
    if fields.get("text") is not None:
        out["text"] = fields["text"]
    post_ip = _first_present(fields, "ip_location", "current_ip_location", "location_ip")
    if post_ip:
        out["location_ip"] = post_ip
    return out


def _media_item(ref: str) -> dict[str, Any]:
    path = _unwrap_ref(ref)
    suffix = Path(path.split("?", 1)[0]).suffix.lower()
    if suffix in IMAGE_MEDIA_EXTENSIONS:
        media_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        media_type = "other"
    return {
        "type": media_type,
        "path": path,
        "source": PLATFORM,
    }


def _coerce_tags(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return _parse_scalar(str(value))


def _extract_xhs_topics(text: str) -> list[str]:
    topics: list[str] = []
    for raw in re.findall(r"#([^#\n\r]+?)#", text or ""):
        topic = raw.replace("[话题]", "").strip()
        if topic and topic not in topics:
            topics.append(topic)
    return topics


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.lower() in {"none", "null"}:
        return None
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
    return stripped


def _first_present(fields: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = fields.get(key)
        if value not in (None, ""):
            return value
    return None


def _looks_like_media_item(value: str) -> bool:
    return value.startswith("- [[") or value.startswith("[[") or value.startswith("- ")


def _unwrap_ref(value: str) -> str:
    value = value.strip()
    if value.startswith("- "):
        value = value[2:].strip()
    if value.startswith("[[") and value.endswith("]]"):
        return value[2:-2].strip()
    return value


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", key)
    key = key.strip("_")
    mapping = {
        "self_description": "self_description",
        "current_ip_location": "current_ip_location",
        "tag_list": "tag_list",
        "create_time": "create_time",
        "last_update_time": "last_update_time",
    }
    return mapping.get(key, key)


def iter_txt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XHS txt user notes to unified ARGUS JSON.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing XHS .txt files.",
    )
    parser.add_argument("--output-dir", default="inputs/argus/json/xhs")
    args = parser.parse_args()

    input_path = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = iter_txt_files(input_path)
    if not txt_files:
        raise SystemExit(f"No .txt files found at {input_path}")

    converted = 0
    for txt_path in txt_files:
        out_path = output_dir / f"{txt_path.stem}.json"
        payload = convert_file(txt_path)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        converted += 1
        print(f"[xhs-json] {txt_path} -> {out_path} posts={len(payload.get('posts') or [])}")

    print(f"[xhs-json] done converted={converted}")


if __name__ == "__main__":
    main()
