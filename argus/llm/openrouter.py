"""OpenAI-compatible router helpers for ARGUS backbone calls."""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


PROVIDER_NAME = os.environ.get("ARGUS_ROUTER_PROVIDER", "openai-compatible")
BASE_URL = os.environ.get("ARGUS_ROUTER_BASE_URL", "")
API_KEY_ENV = "ARGUS_ROUTER_API_KEY"
INVESTIGATOR_MODEL = os.environ.get("ARGUS_INVESTIGATOR_MODEL", os.environ.get("ARGUS_ROUTER_MODEL", "gpt-5.5"))
VERIFIER_MODEL = os.environ.get("ARGUS_VERIFIER_MODEL", os.environ.get("ARGUS_ROUTER_MODEL", "gpt-5.5"))
NARRATIVE_MODEL = os.environ.get("ARGUS_NARRATIVE_MODEL", os.environ.get("ARGUS_ROUTER_MODEL", "gpt-5.5"))

_client: OpenAI | None = None


def make_client(api_key: str | None = None) -> OpenAI:
    """Singleton OpenAI client pointed at the configured compatible router."""
    global _client
    if _client is None:
        if not BASE_URL:
            raise RuntimeError("ARGUS_ROUTER_BASE_URL is not set.")
        resolved_api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not resolved_api_key:
            raise RuntimeError(f"{API_KEY_ENV} is not set.")
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=resolved_api_key,
        )
    return _client


def reasoning_extra_body(enabled: bool) -> dict[str, Any]:
    # OpenRouter/Anthropic reasoning continuation used provider-specific
    # fields. The replacement router is standard OpenAI-compatible by default,
    # so do not send provider-specific reasoning controls unless explicitly
    # requested.
    if not enabled or os.environ.get("ARGUS_ROUTER_REASONING", "0") != "1":
        return {}
    return {"reasoning": {"enabled": True}}


def assistant_message_from_openrouter(message: Any) -> dict[str, Any]:
    """Convert an assistant message back into request format.

    Some routers expose `reasoning_details`; if present, preserve it for
    routers that support continuation. Plain OpenAI-compatible providers will
    simply omit it.
    """
    content = getattr(message, "content", None)
    out: dict[str, Any] = {"role": "assistant", "content": content or None}

    reasoning_details = _message_extra(message, "reasoning_details")
    if reasoning_details is not None:
        out["reasoning_details"] = reasoning_details

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


def _message_extra(message: Any, key: str) -> Any:
    if hasattr(message, key):
        return getattr(message, key)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, dict):
            return data.get(key)
    return None
