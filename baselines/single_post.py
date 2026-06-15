#!/usr/bin/env python
"""Naive baseline: fixed single-post extraction, then user-level aggregation.

This is intentionally not the ARGUS framework. There is no HDI, no CPEG, no
verifier, and no tool loop. Each post is processed independently with a fixed
prompt; the final profile is produced from those per-post findings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from argus.config import load_env
from argus.llm.qwen import chat_vision
from argus.metrics import finish_post, finish_run, start_post, start_run
from baselines.common import (
    chat_baseline_text,
    compact_metadata,
    default_text_model,
    load_json_inputs,
    normalize_model_provider,
    parse_json_object,
    remove_stale_error_file,
    safe_name,
    should_skip_existing_profile,
)


POST_SYSTEM_PROMPT = """You are a simple privacy leakage extractor.

Analyze exactly one social media post together with the user's profile metadata.
Extract privacy-relevant findings only when they are directly supported by the
post text, post metadata, user metadata, or visible images if images are
provided. Do not use external tools and do not rely on other posts.

Return strict JSON:
{
  "post_id": "...",
  "findings": [
    {
      "attribute": "location.home | location.work | education.institution | occupation.company | interest.hobby | ...",
      "value": "specific inferred value",
      "category": "identity | location | education | work | social | lifestyle | temporal | other",
      "evidence": "short evidence from this post",
      "confidence": 0.0
    }
  ]
}
"""

AGGREGATE_SYSTEM_PROMPT = """You aggregate independent per-post privacy findings into one user profile.

Use only the provided findings. Merge duplicates, keep multiple plausible
values when needed, and mark weak unsupported inferences with low confidence.

Return strict JSON:
{
  "user_id": "...",
  "attributes": {
    "attribute.name": {
      "value": "best value or list of values",
      "confidence": 0.0,
      "evidence": [{"post_id": "...", "text": "..."}]
    }
  },
  "summary": "brief natural-language summary"
}
"""


def error_text(*, stage: str, provider: str, model: str | None, error: Exception) -> str:
    return (
        f"[[ERROR]] stage={stage} provider={provider} model={model or ''} "
        f"reason={type(error).__name__}: {error}"
    )


def error_post_result(
    *,
    post_id: str,
    stage: str,
    provider: str,
    model: str | None,
    error: Exception,
) -> dict[str, Any]:
    raw = error_text(stage=stage, provider=provider, model=model, error=error)
    return {
        "post_id": post_id,
        "raw": raw,
        "parse": {
            "post_id": post_id,
            "findings": [],
            "errors": [
                {
                    "stage": stage,
                    "provider": provider,
                    "model": model,
                    "message": raw,
                }
            ],
        },
    }


def error_aggregate_result(
    *,
    user_id: str,
    provider: str,
    model: str | None,
    error: Exception,
) -> dict[str, Any]:
    raw = error_text(
        stage="single_post_aggregate",
        provider=provider,
        model=model,
        error=error,
    )
    return {
        "raw": raw,
        "parse": {
            "user_id": user_id,
            "attributes": {},
            "summary": raw,
            "errors": [
                {
                    "stage": "single_post_aggregate",
                    "provider": provider,
                    "model": model,
                    "message": raw,
                }
            ],
        },
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
    model: str | None,
    error: Exception,
) -> str:
    text = str(error)
    if "[[ERROR]]" in text:
        return text
    return error_text(stage=stage, provider=provider, model=model, error=error)


def analyze_post(
    post: dict[str, Any],
    *,
    user_metadata: dict[str, Any],
    model_provider: str,
    model: str | None,
    max_images: int,
) -> dict[str, Any]:
    post_id = str(post.get("post_id") or "unknown_post")
    media_files = list(post.get("media_files") or [])[:max_images]
    compact_user_metadata = compact_metadata(user_metadata)
    compact_post_metadata = compact_metadata(post.get("metadata"))
    use_qwen_vision = bool(media_files) and normalize_model_provider(model_provider) == "qwen"
    prompt_payload = {
        "user_metadata": compact_user_metadata,
        "post_id": post_id,
        "timestamp": post.get("timestamp"),
        "location_ip": post.get("location_ip"),
        "caption": post.get("caption") or "",
        "post_metadata": compact_post_metadata,
        "image_count": len(media_files),
    }
    if use_qwen_vision:
        prompt_payload["media_files"] = media_files
    prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    user_prompt = f"{POST_SYSTEM_PROMPT}\n\nPOST:\n{prompt}"

    def run_text_only(reason: str | None = None) -> str:
        text_prompt = user_prompt
        if media_files:
            text_prompt += (
                "\n\nNOTE: Image pixels were not analyzed for this post. "
                f"The post has {len(media_files)} image(s)."
            )
        if reason:
            text_prompt += f"\nFallback reason: {reason}"
        return chat_baseline_text(
            [{"role": "user", "content": text_prompt}],
            provider=model_provider,
            model=model,
            role="baseline_single_post_text",
            temperature=0.0,
        )

    start_post(post_id, image_count=len(media_files))
    try:
        if use_qwen_vision:
            try:
                raw = chat_vision(
                    media_files,
                    user_prompt,
                    role="baseline_single_post_vl",
                    max_images=max_images,
                )
            except Exception as exc:
                print(f"[single_post] qwen vision failed post={post_id}; fallback=text error={exc}")
                raw = ""
            if not raw.strip():
                raw = run_text_only("qwen vision returned empty output")
        else:
            raw = run_text_only("text-only baseline model")
    except Exception as exc:
        message = error_text(
            stage="single_post_analysis",
            provider=model_provider,
            model=model,
            error=exc,
        )
        print(f"[single_post] post={post_id} failed after retries; {message}")
        raise RuntimeError(message) from exc
    finally:
        finish_post(post_id, step_count=1, tool_call_count=0)

    parsed = parse_json_object(raw)
    if "findings" not in parsed:
        parsed = {"post_id": post_id, "findings": []}
    parsed.setdefault("post_id", post_id)
    return {"post_id": post_id, "raw": raw, "parse": parsed}


def aggregate_user(
    user_id: str,
    metadata: dict[str, Any],
    per_post: list[dict[str, Any]],
    *,
    model_provider: str,
    model: str | None,
) -> dict[str, Any]:
    findings = [
        {
            "post_id": item["post_id"],
            "findings": item.get("parse", {}).get("findings", []),
        }
        for item in per_post
    ]
    prompt = (
        f"{AGGREGATE_SYSTEM_PROMPT}\n\n"
        f"USER_ID: {user_id}\n"
        f"USER_METADATA:\n{json.dumps(compact_metadata(metadata), ensure_ascii=False, indent=2)}\n\n"
        f"PER_POST_FINDINGS:\n{json.dumps(findings, ensure_ascii=False, indent=2)}"
    )
    raw = chat_baseline_text(
        [{"role": "user", "content": prompt}],
        provider=model_provider,
        model=model,
        role="baseline_single_post_aggregate",
        temperature=0.0,
    )
    parsed = parse_json_object(raw)
    if not parsed:
        parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
    parsed.setdefault("user_id", user_id)
    parsed.setdefault("attributes", {})
    return {"raw": raw, "parse": parsed}


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[single_post] checkpoint unreadable={path} error={exc}; starting user from scratch")
        return []
    per_post = data.get("per_post")
    if not isinstance(per_post, list):
        return []
    return [item for item in per_post if isinstance(item, dict) and item.get("post_id")]


def save_checkpoint(path: Path, per_post: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"per_post": per_post}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def run(
    input_path: str,
    output_dir: str,
    *,
    model_provider: str,
    model: str | None,
    max_images: int,
) -> None:
    load_env()
    model_provider = normalize_model_provider(model_provider)
    model_name = model or default_text_model(model_provider)
    root = Path(output_dir)
    for json_path, user_id, metadata, posts in load_json_inputs(input_path):
        user_dir = root / safe_name(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        out_path = user_dir / "profile.json"
        checkpoint_path = user_dir / "profile.ckpt.json"
        if should_skip_existing_profile(out_path, log_prefix="single_post", user_id=user_id):
            continue

        print(
            f"[single_post] user={user_id} posts={len(posts)} "
            f"provider={model_provider} model={model_name}"
        )
        per_post = load_checkpoint(checkpoint_path)
        completed_post_ids = {str(item.get("post_id")) for item in per_post}
        if per_post:
            print(
                f"[single_post] resume user={user_id} "
                f"completed_posts={len(per_post)} checkpoint={checkpoint_path}"
            )

        start_run(user_id)
        for idx, post in enumerate(posts, start=1):
            post_id = str(post.get("post_id") or f"post_{idx:03d}")
            if post_id in completed_post_ids:
                print(f"[single_post] skip post={post_id} checkpoint")
                continue
            try:
                result = analyze_post(
                    post,
                    user_metadata=metadata,
                    model_provider=model_provider,
                    model=model_name,
                    max_images=max_images,
                )
            except Exception as exc:
                metrics = finish_run()
                error_message = exception_message(
                    stage="single_post_analysis",
                    provider=model_provider,
                    model=model_name,
                    error=exc,
                )
                error_path = write_error_file(user_dir, {
                    "user_id": user_id,
                    "baseline": "single_post",
                    "model_provider": model_provider,
                    "model": model_name,
                    "source_json": str(json_path),
                    "post_id": post_id,
                    "message": error_message,
                    "checkpoint": str(checkpoint_path),
                    "completed_posts": sorted(completed_post_ids),
                    "metrics": metrics.to_dict() if metrics else {},
                })
                print(f"[single_post] wrote error={error_path}; stopping job for inspection")
                raise
            per_post.append(result)
            completed_post_ids.add(str(result.get("post_id")))
            save_checkpoint(checkpoint_path, per_post)

        try:
            aggregate = aggregate_user(
                user_id,
                metadata,
                per_post,
                model_provider=model_provider,
                model=model_name,
            )
        except Exception as exc:
            metrics = finish_run()
            error_message = exception_message(
                stage="single_post_aggregate",
                provider=model_provider,
                model=model_name,
                error=exc,
            )
            error_path = write_error_file(user_dir, {
                "user_id": user_id,
                "baseline": "single_post",
                "model_provider": model_provider,
                "model": model_name,
                "source_json": str(json_path),
                "message": error_message,
                "checkpoint": str(checkpoint_path),
                "completed_posts": sorted(completed_post_ids),
                "metrics": metrics.to_dict() if metrics else {},
            })
            print(f"[single_post] wrote error={error_path}; stopping job for inspection")
            raise
        metrics = finish_run()

        payload = {
            "user_id": user_id,
            "baseline": "single_post",
            "model_provider": model_provider,
            "model": model_name,
            "source_json": str(json_path),
            "profile": aggregate["parse"],
            "raw": {
                "per_post": per_post,
                "aggregation": aggregate["raw"],
            },
            "metrics": metrics.to_dict() if metrics else {},
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        remove_stale_error_file(user_dir)
        print(f"[single_post] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive fixed single-post baseline.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", default="outputs/baselines/single_post")
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
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        model_provider=args.model_provider,
        model=args.model,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
