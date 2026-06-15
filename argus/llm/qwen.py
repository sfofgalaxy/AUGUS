"""DashScope OpenAI-compatible helpers for ARGUS Qwen calls."""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from argus.metrics import (
    LLMCallMetric,
    current_post_id,
    estimate_cost_usd,
    record_llm_call,
    usage_from_response,
)
from argus.path_utils import resolve_local_path
from argus.retry import retry_call
from argus.logging_utils import argus_log


PERCEPTION_VL_MODEL = os.environ.get(
    "ARGUS_QWEN_VL_MODEL",
    os.environ.get("ARGUS_PERCEPTION_VL_MODEL", "qwen3.6-plus"),
)
AMTR_FALLBACK_MODEL = os.environ.get(
    "ARGUS_AMTR_FALLBACK_MODEL",
    os.environ.get("ARGUS_QWEN_TEXT_MODEL", "qwen3.7-max"),
)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


QWEN_PERCEPTION_MAX_SIDE = _int_env("ARGUS_QWEN_PERCEPTION_MAX_SIDE", 768)
QWEN_IMAGE_MAX_BYTES = _int_env("ARGUS_QWEN_IMAGE_MAX_BYTES", 1_800_000)
QWEN_PAYLOAD_MAX_CHARS = _int_env("ARGUS_QWEN_PAYLOAD_MAX_CHARS", 24_000_000)
QWEN_IMAGE_MIN_SIDE = _int_env("ARGUS_QWEN_IMAGE_MIN_SIDE", 224)
QWEN_TEXT_ENABLE_THINKING = _bool_env("ARGUS_QWEN_TEXT_ENABLE_THINKING", True)
QWEN_VISION_ENABLE_THINKING = _bool_env("ARGUS_QWEN_VISION_ENABLE_THINKING", False)

_client: OpenAI | None = None


def make_client(api_key: str | None = None) -> OpenAI:
    global _client
    if _client is None:
        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not resolved_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        _client = OpenAI(
            api_key=resolved_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client


def chat_text(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    role: str = "qwen_text",
    temperature: float | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> str:
    model_name = model or AMTR_FALLBACK_MODEL
    kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    thinking = QWEN_TEXT_ENABLE_THINKING if enable_thinking is None else enable_thinking
    kwargs["extra_body"] = {"enable_thinking": thinking}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    started = time.time()
    argus_log(f"llm start provider=dashscope role={role} model={model_name}")
    try:
        resp = retry_call(
            f"provider=dashscope role={role} model={model_name}",
            lambda: make_client().chat.completions.create(**kwargs),
        )
        elapsed = time.time() - started
        usage = usage_from_response(resp)
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="dashscope",
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
            f"llm done provider=dashscope role={role} "
            f"elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)}"
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="dashscope",
            model=model_name,
            elapsed_seconds=time.time() - started,
            ok=False,
            error=str(exc),
        ))
        argus_log(
            f"llm FAILED provider=dashscope role={role} "
            f"elapsed={time.time() - started:.1f}s error={exc}"
        )
        raise


def chat_vision(
    image_paths: list[str],
    prompt: str,
    *,
    model: str | None = None,
    role: str = "perception_vl",
    max_images: int = 8,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    if not image_paths:
        return ""
    content: list[Any] = [{"type": "text", "text": prompt}]
    attached = 0
    payload_chars = len(prompt)
    for path in image_paths[:max_images]:
        image_url = to_data_url(path)
        if image_url is None:
            argus_log(f"vision skip unreadable image role={role} path={path}")
            continue
        data_chars = len(image_url)
        if payload_chars + data_chars > QWEN_PAYLOAD_MAX_CHARS:
            argus_log(
                f"vision skip oversized image role={role} path={path} "
                f"data_chars={data_chars} payload_chars={payload_chars} "
                f"limit={QWEN_PAYLOAD_MAX_CHARS}"
            )
            continue
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        attached += 1
        payload_chars += data_chars
    if attached == 0:
        argus_log(f"vision payload role={role} images=0 no readable images")
        return ""
    argus_log(f"vision payload role={role} images={attached} chars={payload_chars}")
    try:
        return chat_text(
            [{"role": "user", "content": content}],
            model=model or PERCEPTION_VL_MODEL,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=QWEN_VISION_ENABLE_THINKING,
        )
    except Exception as exc:
        if "base64 data can not be empty" in str(exc):
            argus_log(
                f"vision failed role={role} reason=empty_base64_payload; "
                "returning empty vision result"
            )
            return ""
        raise


def to_data_url(path: str) -> str | None:
    if path.startswith(("http://", "https://", "data:")):
        return path
    p = Path(resolve_local_path(path))
    if not p.exists():
        return None
    encoded = _encode_image_for_qwen(p)
    if encoded is None:
        return None
    data_bytes, mime = encoded
    data = base64.b64encode(data_bytes).decode("ascii")
    return f"data:{mime};base64,{data}"


def _encode_image_for_qwen(path: Path) -> tuple[bytes, str] | None:
    max_side = max(1, QWEN_PERCEPTION_MAX_SIDE)
    max_bytes = max(1, QWEN_IMAGE_MAX_BYTES)
    min_side = max(1, min(QWEN_IMAGE_MIN_SIDE, max_side))
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        width, height = image.size
        longest = max(width, height)
        target_side = min(longest, max_side)
        qualities = (82, 72, 62, 52, 42)
        best: bytes | None = None

        while target_side >= min_side:
            scale = target_side / longest
            if scale < 1:
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                candidate = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                candidate = image
            for quality in qualities:
                buf = io.BytesIO()
                candidate.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if best is None or len(data) < len(best):
                    best = data
                if len(data) <= max_bytes:
                    return data, "image/jpeg"
            target_side = int(target_side * 0.85)

        best_bytes = len(best) if best is not None else 0
        argus_log(
            f"vision skip oversized compressed image path={path} "
            f"bytes={best_bytes} limit={max_bytes}"
        )
        return None
    except Exception as exc:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) <= max_bytes:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return data, mime
        argus_log(
            f"vision skip oversized uncompressed image path={path} "
            f"bytes={len(data)} limit={max_bytes} error={exc}"
        )
        return None
