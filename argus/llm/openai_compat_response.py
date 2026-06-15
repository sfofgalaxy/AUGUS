"""Helpers for slightly non-standard OpenAI-compatible chat responses."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable


def coerce_chat_completion_response(resp: Any) -> Any:
    """Return a ChatCompletion-like object when a router returns SSE text.

    Some OpenAI-compatible routers occasionally return a raw text/event-stream
    body even for non-streaming SDK calls. The OpenAI SDK then surfaces a plain
    string such as ``data: {...}``, which breaks downstream ``resp.choices``
    access. This function reconstructs the small subset of the ChatCompletion
    interface used by ARGUS: ``choices[0].message.content``, optional
    ``message.tool_calls``, and ``usage``.
    """
    if not isinstance(resp, str):
        return resp

    payloads = list(_iter_json_payloads(resp))
    if not payloads:
        return resp

    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    model = ""
    response_id = ""
    created = 0
    finish_reason = None

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        usage = payload.get("usage") or usage
        model = str(payload.get("model") or model)
        response_id = str(payload.get("id") or response_id)
        created = int(payload.get("created") or created or 0)

        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason") or finish_reason
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}

            message_content = message.get("content") if isinstance(message, dict) else None
            delta_content = delta.get("content") if isinstance(delta, dict) else None
            if message_content:
                content_parts.append(str(message_content))
            if delta_content:
                content_parts.append(str(delta_content))

            for tool_call in _as_list(message.get("tool_calls") if isinstance(message, dict) else None):
                _merge_tool_call(tool_calls_by_index, tool_call)
            for tool_call in _as_list(delta.get("tool_calls") if isinstance(delta, dict) else None):
                _merge_tool_call(tool_calls_by_index, tool_call)

    tool_calls = [
        _to_namespace(tool_calls_by_index[index])
        for index in sorted(tool_calls_by_index)
        if _tool_call_has_content(tool_calls_by_index[index])
    ]
    message = SimpleNamespace(
        content="".join(content_parts),
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=response_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[choice],
        usage=usage or {},
    )


def response_preview(resp: Any, *, max_chars: int = 300) -> str:
    try:
        if isinstance(resp, str):
            text = resp
        elif isinstance(resp, (dict, list)):
            text = json.dumps(resp, ensure_ascii=False, default=str)
        else:
            text = repr(resp)
    except Exception:
        text = f"<unprintable {type(resp).__name__}>"
    return text[:max_chars]


def _iter_json_payloads(text: str) -> Iterable[Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            yield json.loads(stripped)
            return
        except json.JSONDecodeError:
            pass

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _merge_tool_call(tool_calls_by_index: dict[int, dict[str, Any]], raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    index = int(raw.get("index") or 0)
    current = tool_calls_by_index.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if raw.get("id"):
        current["id"] = raw["id"]
    if raw.get("type"):
        current["type"] = raw["type"]

    raw_function = raw.get("function") or {}
    if not isinstance(raw_function, dict):
        return
    current_function = current.setdefault("function", {"name": "", "arguments": ""})
    if raw_function.get("name"):
        current_function["name"] += str(raw_function["name"])
    if raw_function.get("arguments"):
        current_function["arguments"] += str(raw_function["arguments"])


def _tool_call_has_content(tool_call: dict[str, Any]) -> bool:
    function = tool_call.get("function") or {}
    return bool(tool_call.get("id") or function.get("name") or function.get("arguments"))


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value
