"""ARGUS runner — single-file and batch entry point."""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from argus.pipeline import ARGUSPipeline, PipelineConfig
from argus.metrics import start_run, finish_run
from argus.config import project_root
from argus.path_utils import is_image_media_ref, normalize_media_ref
from argus.logging_utils import argus_log

logger = logging.getLogger(__name__)


def progress(message: str) -> None:
    if os.environ.get("ARGUS_QUIET", "0") != "1":
        argus_log(message)


def build_argus_app():
    """Build a callable that processes a single user input.

    Returns a thunk: `app(user_id, user_metadata, posts) -> dict`.
    """
    from argus.config import load_env
    from argus.tool_registry import init_tool_registry
    load_env()
    init_tool_registry()

    progress("building pipeline: perception -> investigator -> routing-verifier")
    pipeline = ARGUSPipeline.build_default(config=PipelineConfig())

    def app(user_id: str, user_metadata: dict[str, Any], posts: list[dict[str, Any]]) -> dict[str, Any]:
        start_run(user_id)
        profile, cpeg, logs = pipeline.run_user(user_id, user_metadata, posts)
        metrics = finish_run()
        return {
            "user_id": user_id,
            "profile": asdict(profile),
            "cpeg": cpeg.to_dict(),
            "step_logs": [_serialize_step_log(l) for l in logs],
            "metrics": metrics.to_dict() if metrics else {},
        }

    return app


# ── User input loader (compatible with existing user_notes/{user_id}.txt format) ──

def load_user_notes(path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Parse a `user_notes/{user_id}.txt` file into (user_id, metadata, posts).

    Supports both input formats used in this repo:

    1. New ARGUS block format:
        Basic: <free-form metadata, key:value lines or empty>
        Posts (chronological):
        --- Post {post_id} ---
        Caption: ...
        Timestamp: ...
        Media: path1, path2

    2. Existing run.py / preprocessing format:
        Basic:
        - nickname: ...
        Posts (chronological):
        - post_id: 1
          create time: ...
          title: ...
          text: ...
          media:
            - [[path/to/image.jpg]]
    """
    text = Path(path).read_text(encoding="utf-8")
    user_id = Path(path).stem

    parts = text.split("Posts (chronological):", 1)
    metadata = _parse_basic_block(parts[0]) if len(parts) >= 1 else {}
    posts_raw = parts[1] if len(parts) == 2 else ""

    posts = _parse_posts_block(posts_raw)
    _resolve_post_media_refs(posts, path.parent)
    return user_id, metadata, posts


def _parse_basic_block(text: str) -> dict[str, Any]:
    md: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("basic"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            md[k.strip("- ").strip()] = _unwrap_media_ref(v.strip())
    return md


def _parse_posts_block(text: str) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    media_mode = False

    def flush_current() -> None:
        if current is None:
            return
        if not current.get("caption"):
            current["caption"] = _caption_from_legacy_fields(current)
        posts.append(current)

    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped:
            continue

        if stripped.startswith("--- Post"):
            flush_current()
            pid = stripped.replace("--- Post", "").replace("---", "").strip()
            current = {"post_id": pid, "caption": "", "media_files": [], "timestamp": None}
            media_mode = False
            continue
        if stripped.startswith("- post_id:"):
            flush_current()
            pid = stripped.split(":", 1)[1].strip()
            current = {"post_id": pid, "caption": "", "media_files": [], "timestamp": None}
            media_mode = False
            continue

        if current is None:
            continue

        if stripped.startswith("[[") or stripped.startswith("- [["):
            ref = _unwrap_media_ref(stripped.lstrip("- ").strip())
            if ref and ref != "no media":
                current.setdefault("media_files", []).append(ref)
            continue

        if media_mode and stripped.startswith("- "):
            ref = _unwrap_media_ref(stripped[2:].strip())
            if ref and ref != "no media":
                current.setdefault("media_files", []).append(ref)
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key_norm = key.strip("- ").strip().lower()
            value = value.strip()
            if key_norm == "caption":
                current["caption"] = value
                media_mode = False
            elif key_norm in ("timestamp", "create time", "created time"):
                current["timestamp"] = value
                media_mode = False
            elif key_norm in ("media", "media_files"):
                media_mode = True
                inline_media = [_unwrap_media_ref(m.strip()) for m in value.split(",") if m.strip()]
                current["media_files"].extend([m for m in inline_media if m and m != "no media"])
            elif key_norm in ("location", "ip_location", "current ip location"):
                current["location_ip"] = value
                media_mode = False
            else:
                current[key_norm.replace(" ", "_")] = _unwrap_media_ref(value)
                media_mode = False
        else:
            media_mode = False
            if current.get("caption"):
                current["caption"] += "\n" + stripped
    if current is not None:
        flush_current()
    return posts


def _caption_from_legacy_fields(post: dict[str, Any]) -> str:
    parts = [
        str(post.get("title", "")).strip(),
        str(post.get("text", "")).strip(),
        str(post.get("audio_transcript", "")).strip(),
    ]
    return "\n".join(p for p in parts if p)


def _unwrap_media_ref(value: str) -> str:
    value = value.strip()
    if value.startswith("[[") and value.endswith("]]"):
        value = value[2:-2].strip()
    return normalize_media_ref(value)


def _resolve_post_media_refs(posts: list[dict[str, Any]], input_dir: Path) -> None:
    """Resolve media refs from both project-relative and input-relative files."""
    for post in posts:
        refs = post.get("media_files") or []
        image_refs: list[str] = []
        skipped: list[str] = []
        for ref in refs:
            resolved = _resolve_media_ref(ref, input_dir)
            if is_image_media_ref(resolved):
                image_refs.append(resolved)
            else:
                skipped.append(ref)
        post["media_files"] = image_refs
        if skipped:
            progress(
                f"post={post.get('post_id', 'unknown')} skipped non-image media="
                f"{len(skipped)}"
            )


def _resolve_media_ref(ref: str, input_dir: Path) -> str:
    ref = ref.strip()
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref).expanduser()
    if path.is_absolute():
        return str(path)

    root = project_root()
    candidates = [
        root / path,
        Path.cwd() / path,
        input_dir / path,
        input_dir.parent / path,
        input_dir.parent / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((root / path).resolve())


# ── Serializers ──

def _serialize_step_log(log) -> dict[str, Any]:
    out = asdict(log)
    if log.route is not None:
        out["route"] = asdict(log.route)
    return out


# ── CLI ──

def run_one_file(
    *,
    input_path: Path,
    output_dir: Path,
    app,
) -> Path:
    user_id, metadata, posts = load_user_notes(input_path)
    progress(f"loaded {input_path} as user={user_id}, posts={len(posts)}")
    result = app(user_id, metadata, posts)
    user_dir = output_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    profile_path = user_dir / "profile.json"
    metrics_path = user_dir / "metrics.json"
    profile_payload = {k: v for k, v in result.items() if k != "metrics"}
    profile_path.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(result.get("metrics", {}), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    progress(f"wrote profile={profile_path} metrics={metrics_path}")
    return profile_path


def run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    app,
    resume: bool = False,
    limit: int = 0,
    users: list[str] | None = None,
) -> None:
    txt_files = sorted(input_dir.glob("*.txt"))
    if users:
        txt_files = [p for p in txt_files if any(p.stem.startswith(u) for u in users)]
    if limit > 0:
        txt_files = txt_files[:limit]
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"batch users={len(txt_files)}")
    progress(f"input={input_dir}")
    progress(f"output={output_dir}")
    print("-" * 60)

    succeeded = failed = skipped = 0
    for idx, txt_path in enumerate(txt_files, start=1):
        out_path = output_dir / txt_path.stem / "profile.json"
        if resume and out_path.exists():
            progress(f"[{idx}/{len(txt_files)}] skip user={txt_path.stem} already exists")
            skipped += 1
            continue
        progress(f"[{idx}/{len(txt_files)}] start user={txt_path.stem}")
        t0 = time.time()
        try:
            written = run_one_file(input_path=txt_path, output_dir=output_dir, app=app)
            progress(f"[{idx}/{len(txt_files)}] done user={txt_path.stem} elapsed={time.time() - t0:.1f}s -> {written}")
            succeeded += 1
        except Exception:
            progress(f"[{idx}/{len(txt_files)}] FAILED user={txt_path.stem} elapsed={time.time() - t0:.1f}s")
            traceback.print_exc()
            err_path = output_dir / f"{txt_path.stem}_error.txt"
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            failed += 1
            if os.environ.get("ARGUS_CONTINUE_ON_ERROR", "0") != "1":
                raise

    print("\n" + "=" * 60)
    print(f"Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ARGUS on user notes files.")
    parser.add_argument("--input", help="Path to one user_notes/{user_id}.txt")
    parser.add_argument("--batch", action="store_true", help="Run all .txt files in --input-dir")
    parser.add_argument("--input-dir", help="Directory of user_notes .txt files for batch mode")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON results")
    parser.add_argument("--resume", action="store_true", help="Skip outputs that already exist")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N users in batch mode")
    parser.add_argument("--users", nargs="*", help="Only run users whose filename starts with any listed prefix")
    args = parser.parse_args()

    app = build_argus_app()
    output_dir = Path(args.output_dir)
    if args.batch:
        if not args.input_dir:
            raise SystemExit("Error: --input-dir is required with --batch")
        run_batch(
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            app=app,
            resume=args.resume,
            limit=args.limit,
            users=args.users,
        )
    else:
        if not args.input:
            raise SystemExit("Error: --input is required unless --batch is set")
        out_path = run_one_file(input_path=Path(args.input), output_dir=output_dir, app=app)
        print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
