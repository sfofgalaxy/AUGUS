"""Vertex AI Gemini helpers for ARGUS calls.

This avoids routing ARGUS through the legacy LangChain adapters.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from argus.metrics import (
    LLMCallMetric,
    current_post_id,
    estimate_cost_usd,
    record_llm_call,
    usage_from_gemini_response,
)
from argus.path_utils import resolve_local_path
from argus.retry import retry_call
from argus.logging_utils import argus_log


DELEGATION_VISUAL_MODEL = os.environ.get(
    "ARGUS_DELEGATION_VISUAL_MODEL",
    os.environ.get("DEFAULT_MODEL_GEMINI", "gemini-3.1-pro-preview"),
)

_client: Any | None = None


def make_client():
    global _client
    if _client is not None:
        return _client

    from google import genai

    key_path = os.environ.get("GOOGLE_KEY_PATH", "")
    if not key_path:
        raise RuntimeError("GOOGLE_KEY_PATH is not set")
    if not os.path.exists(key_path):
        raise RuntimeError(f"GOOGLE_KEY_PATH does not exist: {key_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    with open(key_path, "r", encoding="utf-8") as f:
        creds = json.load(f)
    project_id = creds.get("project_id", "")
    if not project_id:
        raise RuntimeError("project_id not found in GOOGLE_KEY_PATH JSON")

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    _client = genai.Client(vertexai=True, project=project_id, location=location)
    return _client


def generate_vision(
    image_paths: list[str],
    prompt: str,
    *,
    model: str | None = None,
    role: str = "gemini_vision",
    temperature: float = 1.0,
    thinking_level: str = "low",
    max_images: int = 8,
) -> str:
    from google.genai import types

    parts: list[Any] = [prompt]
    for path in image_paths[:max_images]:
        part = _image_part(types, path)
        if part is not None:
            parts.append(part)
    if len(parts) == 1:
        return "no readable images provided"

    model_name = model or DELEGATION_VISUAL_MODEL
    started = time.time()
    argus_log(
        f"llm start provider=vertex_gemini role={role} "
        f"model={model_name} images={len(parts) - 1}"
    )
    try:
        response = retry_call(
            f"provider=vertex_gemini role={role} model={model_name}",
            lambda: make_client().models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=max(0.0, min(2.0, temperature)),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=_thinking_level(types, thinking_level),
                    ),
                ),
            ),
        )
        usage = usage_from_gemini_response(response)
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="vertex_gemini",
            model=model_name,
            elapsed_seconds=time.time() - started,
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
            f"llm done provider=vertex_gemini role={role} "
            f"elapsed={time.time() - started:.1f}s tokens={usage.get('total_tokens', 0)}"
        )
        return extract_text(response)
    except Exception as exc:
        record_llm_call(LLMCallMetric(
            post_id=current_post_id(),
            role=role,
            provider="vertex_gemini",
            model=model_name,
            elapsed_seconds=time.time() - started,
            ok=False,
            error=str(exc),
        ))
        argus_log(
            f"llm FAILED provider=vertex_gemini role={role} "
            f"elapsed={time.time() - started:.1f}s error={exc}"
        )
        raise


def extract_text(response: Any) -> str:
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    return (getattr(response, "text", "") or "").strip()


def _image_part(types: Any, path: str):
    if path.startswith(("http://", "https://", "data:")):
        return None
    p = Path(resolve_local_path(path))
    if not p.exists():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)


def _thinking_level(types: Any, level: str):
    mapping = {
        "high": types.ThinkingLevel.HIGH,
        "medium": types.ThinkingLevel.MEDIUM,
        "low": types.ThinkingLevel.LOW,
        "none": types.ThinkingLevel.LOW,
    }
    return mapping.get(level.lower(), types.ThinkingLevel.LOW)
