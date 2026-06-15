#!/usr/bin/env python3
"""Extract representative video frames into ARGUS JSON media files.

The main ARGUS pipeline consumes image-like media. Run this preprocessing
script when your unified JSON contains local video files. It writes a new JSON
copy whose `media_files` contain the original images plus extracted JPEG
frames from each video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argus.path_utils import IMAGE_MEDIA_EXTENSIONS

VIDEO_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
}


def iter_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.json") if p.is_file())


def process_json_file(
    json_path: Path,
    *,
    output_dir: Path,
    frame_root: Path,
    frames_per_video: int,
    max_side: int,
    jpeg_quality: int,
) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    user_id = _user_id(payload, json_path)
    user_frame_root = frame_root / user_id

    for post_idx, post in enumerate(payload.get("posts") or [], start=1):
        if not isinstance(post, dict):
            continue
        post_id = str(post.get("post_id") or post.get("id") or f"post_{post_idx:03d}")
        media_items = _media_items(post)
        rewritten: list[dict[str, Any]] = []
        extracted: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        for item in media_items:
            media_type, ref, source = _media_type_ref_source(item)
            if not ref:
                continue
            if _is_video(media_type, ref):
                video_path = _resolve_local_path(ref, json_path.parent)
                if video_path is None:
                    skipped.append({"path": ref, "reason": "remote_or_missing_video"})
                    continue
                out_dir = user_frame_root / post_id / _safe_stem(video_path.stem)
                frame_paths = extract_frames(
                    video_path,
                    out_dir=out_dir,
                    frames_per_video=frames_per_video,
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                )
                if not frame_paths:
                    skipped.append({"path": str(video_path), "reason": "no_frames_extracted"})
                    continue
                for frame_idx, frame_path in enumerate(frame_paths, start=1):
                    frame_item = {
                        "type": "image",
                        "path": str(frame_path),
                        "source": "video_frame",
                        "metadata": {
                            "source_video": str(video_path),
                            "source": source,
                            "frame_index": frame_idx,
                        },
                    }
                    rewritten.append(frame_item)
                    extracted.append(frame_item)
            elif _is_image(media_type, ref):
                rewritten.append(_normalize_image_item(item, ref, source))
            else:
                skipped.append({"path": ref, "reason": "unsupported_media_type"})

        post["media_files"] = rewritten
        metadata = post.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["video_frame_extraction"] = {
                "frames_per_video": frames_per_video,
                "max_side": max_side,
                "extracted_count": len(extracted),
                "skipped": skipped,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / json_path.name
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def extract_frames(
    video_path: Path,
    *,
    out_dir: Path,
    frames_per_video: int,
    max_side: int,
    jpeg_quality: int,
) -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        positions = list(range(max(1, frames_per_video)))
    else:
        sample_count = max(1, min(frames_per_video, total_frames))
        positions = [
            int(round((i + 1) * (total_frames - 1) / (sample_count + 1)))
            for i in range(sample_count)
        ]

    written: list[Path] = []
    for idx, position in enumerate(positions, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _resize_frame(frame, max_side=max_side)
        out_path = out_dir / f"frame_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        written.append(out_path.resolve())

    cap.release()
    return written


def _resize_frame(frame: Any, *, max_side: int):
    if max_side <= 0:
        return frame
    import cv2

    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _media_items(post: dict[str, Any]) -> list[Any]:
    raw = (
        post.get("media_files")
        or post.get("media")
        or post.get("images")
        or post.get("image_files")
        or []
    )
    if isinstance(raw, (str, dict)):
        return [raw]
    return list(raw or [])


def _media_type_ref_source(item: Any) -> tuple[str | None, str, str | None]:
    if isinstance(item, str):
        return None, item, None
    if isinstance(item, dict):
        media_type = item.get("type") or item.get("media_type") or item.get("kind")
        ref = item.get("path") or item.get("file") or item.get("url") or item.get("src") or ""
        source = item.get("source")
        return str(media_type).lower() if media_type else None, str(ref), str(source) if source else None
    return None, "", None


def _normalize_image_item(item: Any, ref: str, source: str | None) -> dict[str, Any]:
    if isinstance(item, dict):
        out = dict(item)
        out.setdefault("type", "image")
        out.setdefault("path", ref)
        if source:
            out.setdefault("source", source)
        return out
    return {"type": "image", "path": ref, "source": source or "input"}


def _is_video(media_type: str | None, ref: str) -> bool:
    if media_type == "video":
        return True
    return Path(ref.split("?", 1)[0]).suffix.lower() in VIDEO_MEDIA_EXTENSIONS


def _is_image(media_type: str | None, ref: str) -> bool:
    if media_type in {"image", "photo", None}:
        suffix = Path(ref.split("?", 1)[0]).suffix.lower()
        return suffix in IMAGE_MEDIA_EXTENSIONS or ref.startswith(("http://", "https://", "data:image/"))
    return False


def _resolve_local_path(ref: str, base_dir: Path) -> Path | None:
    if ref.startswith(("http://", "https://", "data:")):
        return None
    path = Path(ref).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, PROJECT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _user_id(payload: dict[str, Any], json_path: Path) -> str:
    user = payload.get("user") or {}
    return str(
        user.get("user_id")
        or payload.get("user_id")
        or user.get("username")
        or json_path.stem
    )


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80] or "video"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract local video frames for ARGUS JSON inputs.")
    parser.add_argument("--input", required=True, help="Input user JSON file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for rewritten JSON files.")
    parser.add_argument(
        "--frame-dir",
        default=None,
        help="Directory for extracted frames. Defaults to <output-dir>/_frames.",
    )
    parser.add_argument("--frames-per-video", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    frame_root = Path(args.frame_dir).expanduser() if args.frame_dir else output_dir / "_frames"

    written = []
    for json_path in iter_json_files(input_path):
        out_path = process_json_file(
            json_path,
            output_dir=output_dir,
            frame_root=frame_root,
            frames_per_video=args.frames_per_video,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
        )
        written.append(out_path)
        print(f"[video-frames] {json_path} -> {out_path}")

    print(f"[video-frames] done files={len(written)} output={output_dir} frames={frame_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
