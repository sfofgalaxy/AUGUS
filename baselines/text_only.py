#!/usr/bin/env python
"""Text-only baseline: one user-level pass over all posts.

This baseline receives user metadata and all post text/metadata at once. It
does not receive image pixels, does not call tools, and does not run per-post
loops.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from argus.config import load_env
from argus.metrics import finish_run, start_run
from baselines.common import (
    chat_baseline_text,
    compact_metadata,
    compact_posts_for_text,
    default_text_model,
    int_env,
    load_json_inputs,
    normalize_model_provider,
    parse_json_object,
    remove_stale_error_file,
    safe_name,
    should_skip_existing_profile,
)


TEXT_ONLY_SYSTEM_PROMPT = """You are a text-only privacy leakage baseline.

You receive one user's profile metadata and all of their posts as text and
structured metadata. You do not see image pixels and you cannot call tools.

Use only the provided text and metadata. Do not invent image observations. Treat
user-level metadata and post-level metadata separately:
- user tags, such as XHS profile tag_list, describe the user profile.
- post hashtags, topics, tagged users, timestamps, and locations are evidence
  for that specific post, not user-level tags by themselves.

Return strict JSON:
{
  "user_id": "...",
  "attributes": {
    "attribute.name": {
      "value": "specific inferred value",
      "confidence": 0.0,
      "evidence": [{"post_id": "...", "text": "...", "source": "text|metadata|timestamp|location"}]
    }
  },
  "summary": "brief natural-language summary"
}

Only include findings that are supported by the supplied text or metadata. Weak
or ambiguous observations can be included with low confidence, but do not turn
one-off objects or post hashtags into stable user traits unless the evidence is
clear.
"""


def error_text(*, stage: str, provider: str, model: str, error: Exception) -> str:
    return (
        f"[[ERROR]] stage={stage} provider={provider} model={model} "
        f"reason={type(error).__name__}: {error}"
    )


def error_profile(
    *,
    user_id: str,
    stage: str,
    provider: str,
    model: str,
    error: Exception,
) -> tuple[str, dict[str, Any]]:
    raw = error_text(stage=stage, provider=provider, model=model, error=error)
    return raw, {
        "user_id": user_id,
        "attributes": {},
        "summary": raw,
        "errors": [
            {
                "stage": stage,
                "provider": provider,
                "model": model,
                "message": raw,
            }
        ],
    }


def write_error_file(user_dir: Path, payload: dict[str, Any]) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    error_path = user_dir / "error.json"
    error_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return error_path


def exception_message(
    *,
    stage: str,
    provider: str,
    model: str,
    error: Exception,
) -> str:
    text = str(error)
    if "[[ERROR]]" in text:
        return text
    return error_text(stage=stage, provider=provider, model=model, error=error)


def build_user_prompt(user_id: str, metadata: dict[str, Any], posts: list[dict[str, Any]]) -> str:
    return build_prompt_from_compact_posts(
        user_id,
        metadata,
        compact_posts_for_text(posts),
        task_note="Analyze all posts together and return the final user profile.",
    )


def build_prompt_from_compact_posts(
    user_id: str,
    metadata: dict[str, Any],
    compact_posts: list[dict[str, Any]],
    *,
    task_note: str,
) -> str:
    payload = {
        "user_id": user_id,
        "user_metadata": compact_metadata(metadata),
        "posts": compact_posts,
    }
    return (
        f"{TEXT_ONLY_SYSTEM_PROMPT}\n\n"
        f"TASK_NOTE: {task_note}\n\n"
        f"USER_TEXT_ONLY_INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_merge_prompt(
    user_id: str,
    metadata: dict[str, Any],
    chunk_profiles: list[dict[str, Any]],
) -> str:
    payload = {
        "user_id": user_id,
        "user_metadata": compact_metadata(metadata),
        "chunk_profiles": chunk_profiles,
    }
    return (
        f"{TEXT_ONLY_SYSTEM_PROMPT}\n\n"
        "TASK_NOTE: Merge these chunk-level text-only profiles into one final user profile. "
        "Deduplicate repeated attributes, preserve evidence post_ids, and do not invent "
        "anything not supported by the chunks.\n\n"
        f"CHUNK_PROFILES:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def provider_prompt_limit(model_provider: str, model_name: str) -> int:
    provider = normalize_model_provider(model_provider)
    if provider == "qwen":
        # DashScope qwen3.7-max rejects very long inputs with:
        # "Range of input length should be [1, 229376]". Character count is only
        # an approximation of token length, so keep the default comfortably below
        # the service limit. Set BASELINE_TEXT_ONLY_QWEN_MAX_PROMPT_CHARS=0 to
        # disable chunking explicitly.
        return int_env("BASELINE_TEXT_ONLY_QWEN_MAX_PROMPT_CHARS", 120_000)
    return int_env("BASELINE_TEXT_ONLY_MAX_PROMPT_CHARS", 0)


def make_post_chunks(
    user_id: str,
    metadata: dict[str, Any],
    posts: list[dict[str, Any]],
    *,
    max_prompt_chars: int,
) -> list[list[dict[str, Any]]]:
    compact_posts = compact_posts_for_text(posts)
    if max_prompt_chars <= 0:
        return [compact_posts]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for post in compact_posts:
        candidate = current + [post]
        prompt = build_prompt_from_compact_posts(
            user_id,
            metadata,
            candidate,
            task_note="Analyze this chunk of posts only. Return a chunk-level user profile.",
        )
        if current and len(prompt) > max_prompt_chars:
            chunks.append(current)
            current = [post]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_chunk_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[text_only] checkpoint unreadable={path} error={exc}; starting user from scratch")
        return []
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [item for item in chunks if isinstance(item, dict) and item.get("post_ids")]


def save_chunk_checkpoint(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def run_chunked_user(
    user_id: str,
    metadata: dict[str, Any],
    posts: list[dict[str, Any]],
    *,
    model_provider: str,
    model_name: str,
    checkpoint_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    max_prompt_chars = provider_prompt_limit(model_provider, model_name)
    chunks = make_post_chunks(
        user_id,
        metadata,
        posts,
        max_prompt_chars=max_prompt_chars,
    )
    if len(chunks) <= 1:
        try:
            raw = chat_baseline_text(
                [{"role": "user", "content": build_user_prompt(user_id, metadata, posts)}],
                provider=model_provider,
                model=model_name,
                role="baseline_text_only",
                temperature=0.0,
            )
            parsed = parse_json_object(raw)
        except Exception as exc:
            message = error_text(
                stage="baseline_text_only",
                provider=model_provider,
                model=model_name,
                error=exc,
            )
            print(f"[text_only] user={user_id} failed after retries; {message}")
            raise RuntimeError(message) from exc
        return raw, parsed, {"mode": "single", "chunks": []}

    print(
        f"[text_only] chunked user={user_id} chunks={len(chunks)} "
        f"max_prompt_chars={max_prompt_chars}"
    )
    checkpoint_chunks = load_chunk_checkpoint(checkpoint_path)
    completed = {
        tuple(str(pid) for pid in item.get("post_ids", [])): item
        for item in checkpoint_chunks
    }
    raw_chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        post_ids = [str(post.get("post_id")) for post in chunk]
        key = tuple(post_ids)
        if key in completed:
            print(f"[text_only] skip chunk={idx}/{len(chunks)} checkpoint posts={len(post_ids)}")
            raw_chunks.append(completed[key])
            continue
        prompt = build_prompt_from_compact_posts(
            user_id,
            metadata,
            chunk,
            task_note=(
                f"Analyze chunk {idx}/{len(chunks)} only. Return a chunk-level "
                "profile using evidence from these posts."
            ),
        )
        print(
            f"[text_only] run chunk={idx}/{len(chunks)} "
            f"posts={len(post_ids)} prompt_chars={len(prompt)}"
        )
        try:
            raw = chat_baseline_text(
                [{"role": "user", "content": prompt}],
                provider=model_provider,
                model=model_name,
                role="baseline_text_only_chunk",
                temperature=0.0,
            )
            parsed = parse_json_object(raw)
        except Exception as exc:
            message = error_text(
                stage="baseline_text_only_chunk",
                provider=model_provider,
                model=model_name,
                error=exc,
            )
            print(
                f"[text_only] user={user_id} chunk={idx}/{len(chunks)} "
                f"failed after retries; {message}"
            )
            raise RuntimeError(message) from exc
        if not parsed:
            parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
        item = {
            "chunk_index": idx,
            "post_ids": post_ids,
            "raw": raw,
            "parse": parsed,
        }
        raw_chunks.append(item)
        save_chunk_checkpoint(checkpoint_path, raw_chunks)

    merge_prompt = build_merge_prompt(
        user_id,
        metadata,
        [
            {
                "chunk_index": item.get("chunk_index"),
                "post_ids": item.get("post_ids"),
                "profile": item.get("parse"),
            }
            for item in raw_chunks
        ],
    )
    print(f"[text_only] merge chunks={len(raw_chunks)} prompt_chars={len(merge_prompt)}")
    try:
        raw = chat_baseline_text(
            [{"role": "user", "content": merge_prompt}],
            provider=model_provider,
            model=model_name,
            role="baseline_text_only_merge",
            temperature=0.0,
        )
        parsed = parse_json_object(raw)
    except Exception as exc:
        message = error_text(
            stage="baseline_text_only_merge",
            provider=model_provider,
            model=model_name,
            error=exc,
        )
        print(f"[text_only] user={user_id} merge failed after retries; {message}")
        raise RuntimeError(message) from exc
    return raw, parsed, {"mode": "chunked", "chunks": raw_chunks}


def run(
    input_path: str,
    output_dir: str,
    *,
    model_provider: str,
    model: str | None,
) -> None:
    load_env()
    model_provider = normalize_model_provider(model_provider)
    model_name = model or default_text_model(model_provider)
    root = Path(output_dir)
    for json_path, user_id, metadata, posts in load_json_inputs(input_path):
        user_dir = root / safe_name(user_id)
        out_path = user_dir / "profile.json"
        checkpoint_path = user_dir / "profile.ckpt.json"
        if should_skip_existing_profile(out_path, log_prefix="text_only", user_id=user_id):
            continue

        print(
            f"[text_only] user={user_id} posts={len(posts)} "
            f"provider={model_provider} model={model_name}"
        )
        start_run(user_id)
        try:
            raw, parsed, run_details = run_chunked_user(
                user_id,
                metadata,
                posts,
                model_provider=model_provider,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            metrics = finish_run()
            message = exception_message(
                stage="baseline_text_only",
                provider=model_provider,
                model=model_name,
                error=exc,
            )
            error_path = write_error_file(user_dir, {
                "user_id": user_id,
                "baseline": "text_only",
                "model_provider": model_provider,
                "model": model_name,
                "source_json": str(json_path),
                "message": message,
                "checkpoint": str(checkpoint_path),
                "metrics": metrics.to_dict() if metrics else {},
            })
            print(f"[text_only] wrote error={error_path}; stopping job for inspection")
            raise
        metrics = finish_run()
        if not parsed:
            parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
        parsed.setdefault("user_id", user_id)
        parsed.setdefault("attributes", {})

        payload = {
            "user_id": user_id,
            "baseline": "text_only",
            "model_provider": model_provider,
            "model": model_name,
            "source_json": str(json_path),
            "profile": parsed,
            "raw_output": raw,
            "run_details": run_details,
            "metrics": metrics.to_dict() if metrics else {},
        }
        user_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        remove_stale_error_file(user_dir)
        print(f"[text_only] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-pass text-only JSON baseline.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", default="outputs/baselines/text_only")
    parser.add_argument(
        "--model-provider",
        default=os.environ.get("BASELINE_MODEL_PROVIDER", "gpt"),
        choices=["gpt", "qwen", "gemini", "shadowapi", "dashscope", "vertex_gemini"],
        help="Text model provider. gpt uses ARGUS_ROUTER_API_KEY / shadowapi.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BASELINE_MODEL"),
        help="Model name. Defaults to gpt-5.5, qwen3.7-max, or gemini-3.1-pro-preview.",
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        model_provider=args.model_provider,
        model=args.model,
    )


if __name__ == "__main__":
    main()
