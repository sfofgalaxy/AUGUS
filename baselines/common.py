"""Shared helpers for JSON baselines."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from argus.json_input import iter_user_json_files, load_user_json
from argus.logging_utils import argus_log
from argus.llm.openai_compat_response import (
    coerce_chat_completion_response,
    response_preview,
)
from argus.metrics import (
    LLMCallMetric,
    current_post_id,
    estimate_cost_usd,
    record_llm_call,
    usage_from_gemini_response,
    usage_from_response,
)
from argus.retry import retry_call


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def load_json_inputs(input_path: str | Path) -> list[tuple[Path, str, dict[str, Any], list[dict[str, Any]]]]:
    loaded = []
    for path in iter_user_json_files(input_path):
        user_id, metadata, posts = load_user_json(path)
        loaded.append((path, user_id, metadata, posts))
    if not loaded:
        raise SystemExit(f"No .json input files found at {input_path}")
    return loaded


def write_json_result(result: dict[str, Any], user_dir: Path) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    profile_path = user_dir / "profile.json"
    metrics_path = user_dir / "metrics.json"
    payload = {k: v for k, v in result.items() if k != "metrics"}
    profile_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(result.get("metrics", {}), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return profile_path


def has_error_marker(path: Path) -> bool:
    """Return True when an existing output is an error placeholder."""
    try:
        return "[[ERROR]]" in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def should_skip_existing_profile(
    out_path: Path,
    *,
    log_prefix: str,
    user_id: str,
) -> bool:
    """Skip only successful existing profiles.

    Older runs may have written a profile.json containing ``[[ERROR]]`` after
    exhausting retries. Those should be retried on the next run instead of
    being treated as complete.
    """
    if not out_path.exists():
        return False
    if has_error_marker(out_path):
        print(f"[{log_prefix}] retry user={user_id} existing error marker={out_path}")
        return False
    print(f"[{log_prefix}] skip user={user_id} existing={out_path}")
    return True


def remove_stale_error_file(user_dir: Path) -> None:
    error_path = user_dir / "error.json"
    if error_path.exists():
        try:
            error_path.unlink()
        except OSError:
            pass


def attr_score(info: dict[str, Any]) -> tuple[float, float, int]:
    confidence = float(info.get("confidence") or 0.0)
    support = float(info.get("support_score") or 0.0)
    granularity = float(info.get("granularity_score") or 0.0)
    evidence_count = len(info.get("evidence_ids") or [])
    return (confidence * (1.0 + support) ** 0.5, granularity, evidence_count)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_model_provider(provider: str | None) -> str:
    value = (provider or "gpt").strip().lower()
    if value in {"gpt", "shadowapi", "router", "openrouter"}:
        return "gpt"
    if value in {"qwen", "dashscope"}:
        return "qwen"
    if value in {"gemini", "vertex", "vertex_gemini"}:
        return "gemini"
    raise ValueError(f"Unsupported model provider: {provider!r}. Use gpt, qwen, or gemini.")


def default_text_model(provider: str) -> str:
    provider = normalize_model_provider(provider)
    if provider == "gpt":
        return os.environ.get(
            "ARGUS_BASELINE_GPT_MODEL",
            os.environ.get("ARGUS_ROUTER_MODEL", "gpt-5.5"),
        )
    if provider == "gemini":
        return os.environ.get(
            "ARGUS_BASELINE_GEMINI_MODEL",
            os.environ.get("DEFAULT_MODEL_GEMINI", "gemini-3.1-pro-preview"),
        )
    return os.environ.get(
        "ARGUS_BASELINE_QWEN_TEXT_MODEL",
        os.environ.get("ARGUS_QWEN_TEXT_MODEL", "qwen3.7-max"),
    )


def default_multimodal_model(provider: str) -> str:
    provider = normalize_model_provider(provider)
    if provider == "gpt":
        return os.environ.get(
            "ARGUS_BASELINE_GPT_VISION_MODEL",
            os.environ.get("ARGUS_BASELINE_GPT_MODEL", os.environ.get("ARGUS_ROUTER_MODEL", "gpt-5.5")),
        )
    if provider == "gemini":
        return os.environ.get(
            "ARGUS_BASELINE_GEMINI_MODEL",
            os.environ.get("DEFAULT_MODEL_GEMINI", "gemini-3.1-pro-preview"),
        )
    return os.environ.get(
        "ARGUS_BASELINE_QWEN_VISION_MODEL",
        os.environ.get("ARGUS_QWEN_VL_MODEL", os.environ.get("ARGUS_PERCEPTION_VL_MODEL", "qwen3.6-plus")),
    )


def int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def chat_baseline_text(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None = None,
    role: str,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
) -> str:
    provider = normalize_model_provider(provider)
    model_name = model or default_text_model(provider)
    if provider == "qwen":
        from argus.llm.qwen import chat_text as qwen_chat_text

        return qwen_chat_text(
            messages,
            model=model_name,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "gemini":
        return _chat_gemini_text(
            messages,
            model=model_name,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    from argus.llm.openrouter import PROVIDER_NAME, make_client

    kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    started = time.time()
    argus_log(f"llm start provider={PROVIDER_NAME} role={role} model={model_name}")
    try:
        label = f"provider={PROVIDER_NAME} role={role} model={model_name}"
        resp = retry_call(
            label,
            lambda: _create_valid_gpt_completion(make_client, kwargs, label=label),
        )
        elapsed = time.time() - started
        usage = usage_from_response(resp)
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider=PROVIDER_NAME,
            model=model_name,
            elapsed_seconds=elapsed,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            estimated_cost_usd=estimate_cost_usd(
                model_name,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            ),
            ok=True,
        ))
        argus_log(
            f"llm done provider={PROVIDER_NAME} role={role} "
            f"elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)}"
        )
        return _gpt_completion_content(resp)
    except Exception as exc:
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider=PROVIDER_NAME,
            model=model_name,
            elapsed_seconds=time.time() - started,
            ok=False,
            error=str(exc),
        ))
        argus_log(
            f"llm FAILED provider={PROVIDER_NAME} role={role} "
            f"elapsed={time.time() - started:.1f}s error={exc}"
        )
        raise


def chat_baseline_vision(
    image_paths: list[str],
    prompt: str,
    *,
    provider: str,
    model: str | None = None,
    role: str,
    max_images: int,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Call one configured multimodal model with images.

    Prior-work baselines use this to keep visual extraction and text reasoning
    on the same model family/model instead of silently mixing in ARGUS
    specialist models.
    """
    provider = normalize_model_provider(provider)
    model_name = model or default_multimodal_model(provider)
    selected = image_paths if max_images < 0 else image_paths[:max_images]
    if not selected:
        return chat_baseline_text(
            [{"role": "user", "content": prompt}],
            provider=provider,
            model=model_name,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "qwen":
        from argus.llm.qwen import chat_vision as qwen_chat_vision

        return qwen_chat_vision(
            selected,
            prompt,
            model=model_name,
            role=role,
            max_images=len(selected),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        from argus.llm.gemini import generate_vision

        return generate_vision(
            selected,
            prompt,
            model=model_name,
            role=role,
            temperature=0.0 if temperature is None else temperature,
            thinking_level="low",
            max_images=len(selected),
        )

    return _chat_gpt_vision(
        selected,
        prompt,
        model=model_name,
        role=role,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _chat_gpt_vision(
    image_paths: list[str],
    prompt: str,
    *,
    model: str,
    role: str,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    from argus.llm.openrouter import PROVIDER_NAME, make_client
    from argus.llm.qwen import to_data_url

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    attached = 0
    payload_chars = len(prompt)
    for path in image_paths:
        image_url = to_data_url(path)
        if image_url is None:
            argus_log(f"vision skip unreadable image role={role} path={path}")
            continue
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        payload_chars += len(image_url)
        attached += 1

    if attached == 0:
        return chat_baseline_text(
            [{"role": "user", "content": prompt}],
            provider="gpt",
            model=model,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    started = time.time()
    argus_log(
        f"llm start provider={PROVIDER_NAME} role={role} model={model} "
        f"images={attached} chars={payload_chars}"
    )
    try:
        label = f"provider={PROVIDER_NAME} role={role} model={model}"
        resp = retry_call(
            label,
            lambda: _create_valid_gpt_completion(make_client, kwargs, label=label),
        )
        elapsed = time.time() - started
        usage = usage_from_response(resp)
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider=PROVIDER_NAME,
            model=model,
            elapsed_seconds=elapsed,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            estimated_cost_usd=estimate_cost_usd(
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            ),
            ok=True,
            metadata={"image_count": attached},
        ))
        argus_log(
            f"llm done provider={PROVIDER_NAME} role={role} "
            f"elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)}"
        )
        return _gpt_completion_content(resp)
    except Exception as exc:
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider=PROVIDER_NAME,
            model=model,
            elapsed_seconds=time.time() - started,
            ok=False,
            error=str(exc),
            metadata={"image_count": attached},
        ))
        argus_log(
            f"llm FAILED provider={PROVIDER_NAME} role={role} "
            f"elapsed={time.time() - started:.1f}s error={exc}"
        )
        raise


def _create_valid_gpt_completion(make_client_fn, kwargs: dict[str, Any], *, label: str):
    resp = coerce_chat_completion_response(make_client_fn().chat.completions.create(**kwargs))
    _ensure_valid_gpt_completion(resp, label=label)
    return resp


def _ensure_valid_gpt_completion(resp: Any, *, label: str) -> None:
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(
            f"invalid OpenAI-compatible response for {label}: "
            f"missing choices; type={type(resp).__name__}; preview={response_preview(resp)}"
        )
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError(
            f"invalid OpenAI-compatible response for {label}: "
            f"missing choices[0].message; preview={response_preview(resp)}"
        )


def _gpt_completion_content(resp: Any) -> str:
    return getattr(resp.choices[0].message, "content", None) or ""


def _chat_gemini_text(
    messages: list[dict[str, Any]],
    *,
    model: str,
    role: str,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    from google.genai import types

    from argus.llm.gemini import extract_text, make_client

    system_parts: list[str] = []
    content_parts: list[str] = []
    for message in messages:
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if message.get("role") == "system":
            system_parts.append(content)
        else:
            content_parts.append(content)

    config_kwargs: dict[str, Any] = {
        "temperature": 0.0 if temperature is None else max(0.0, min(2.0, temperature)),
        "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    }
    if system_parts:
        config_kwargs["system_instruction"] = "\n\n".join(system_parts)
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    started = time.time()
    argus_log(f"llm start provider=vertex_gemini role={role} model={model}")
    try:
        response = retry_call(
            f"provider=vertex_gemini role={role} model={model}",
            lambda: make_client().models.generate_content(
                model=model,
                contents="\n\n".join(content_parts),
                config=types.GenerateContentConfig(**config_kwargs),
            ),
        )
        elapsed = time.time() - started
        usage = usage_from_gemini_response(response)
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="vertex_gemini",
            model=model,
            elapsed_seconds=elapsed,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            estimated_cost_usd=estimate_cost_usd(
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            ),
            ok=True,
        ))
        argus_log(
            f"llm done provider=vertex_gemini role={role} "
            f"elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)}"
        )
        return extract_text(response)
    except Exception as exc:
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="vertex_gemini",
            model=model,
            elapsed_seconds=time.time() - started,
            ok=False,
            error=str(exc),
        ))
        argus_log(
            f"llm FAILED provider=vertex_gemini role={role} "
            f"elapsed={time.time() - started:.1f}s error={exc}"
        )
        raise


def compact_metadata(metadata: dict[str, Any] | None, *, max_value_chars: int = 1200) -> dict[str, Any]:
    """Keep prompt-useful metadata while dropping large raw platform payloads."""
    if not metadata:
        return {}
    skip = {
        "raw",
        "info",
        "meta",
        "image_versions2",
        "resources",
        "clips_metadata",
        "profile_pic_url",
        "profile_pic_url_hd",
        "thumbnail_url",
        "input_json",
        "source_dir",
        "post_dir",
    }
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in skip or value in (None, "", [], {}):
            continue
        compact[key] = _compact_value(value, max_value_chars=max_value_chars)
    return compact


def compact_posts_for_text(
    posts: list[dict[str, Any]],
    *,
    max_caption_chars: int | None = None,
) -> list[dict[str, Any]]:
    if max_caption_chars is None:
        max_caption_chars = int_env("BASELINE_POST_TEXT_MAX_CHARS", 4000)
    compact_posts: list[dict[str, Any]] = []
    for post in posts:
        compact_posts.append({
            "post_id": post.get("post_id"),
            "timestamp": post.get("timestamp"),
            "location_ip": post.get("location_ip"),
            "caption": truncate_text(post.get("caption") or "", max_caption_chars),
            "metadata": compact_metadata(post.get("metadata")),
            "image_count": len(post.get("media_files") or []),
        })
    return compact_posts


def truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _compact_value(value: Any, *, max_value_chars: int) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if v in (None, "", [], {}):
                continue
            out[str(k)] = _compact_value(v, max_value_chars=max_value_chars)
        return out
    if isinstance(value, list):
        return [_compact_value(v, max_value_chars=max_value_chars) for v in value[:20]]
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        if len(text) > max_value_chars:
            return text[:max_value_chars] + "...[truncated]"
        return value
    text = str(value)
    if len(text) > max_value_chars:
        return text[:max_value_chars] + "...[truncated]"
    return text
