"""OpenAI-compatible backbone router wiring.

ARGUS uses an OpenAI-compatible router for backbone text/tool calls:
  - Investigator (backbone, tool calling)  : GPT-5.5 by default
  - Routing-Verifier (optional, no tools)  : GPT-5.5 when selected
  - Profile narrative (Layer 2)            : GPT-5.5 by default

Gemini and Qwen calls go through their **native APIs** (see argus/adapters.py):
  - Gemini → Vertex AI (GOOGLE_KEY_PATH service account)
  - Qwen   → DashScope (DASHSCOPE_API_KEY, OpenAI-compat endpoint)

The investigator's visual delegation tool (`deep_visual_analysis`) is
**passed in** by the caller and implemented in adapters.py against the native
Gemini API.

Investigator does NOT directly receive raw images; it only sees the
PerceptualSignature. Original images flow only when the Investigator
explicitly invokes `deep_visual_analysis`.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from argus.metrics import (
    LLMCallMetric,
    current_post_id,
    estimate_cost_usd,
    record_llm_call,
    usage_from_response,
)
from argus.llm.openrouter import (
    INVESTIGATOR_MODEL,
    NARRATIVE_MODEL,
    PROVIDER_NAME,
    VERIFIER_MODEL,
    assistant_message_from_openrouter,
    make_client,
    reasoning_extra_body,
)
from argus.llm.openai_compat_response import (
    coerce_chat_completion_response,
    response_preview,
)
from argus.retry import retry_call
from argus.logging_utils import argus_log

logger = logging.getLogger(__name__)


def _create_valid_chat_completion(client, *, label: str, **kwargs):
    resp = coerce_chat_completion_response(client.chat.completions.create(**kwargs))
    _ensure_valid_chat_completion(resp, label=label)
    return resp


def _ensure_valid_chat_completion(resp: Any, *, label: str) -> None:
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


# ── Delegation tool schema (used by ToolDispatcher when wired in) ──
#
# Implementations live in argus/adapters.py (Gemini via Vertex AI). This
# file only declares the schemas so the backbone tool-call interface knows
# they exist.

def delegation_schema(name: str) -> dict:
    if name == "deep_visual_analysis":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    "Delegate deep visual analysis of original post images to "
                    "a vision-specialist model (Gemini 3.1 Pro). Use this when "
                    "the perceptual signature suggests there is more visual "
                    "detail to extract — e.g. fine text on documents, "
                    "small landmarks, navigation maps, blurry signage. Ask Gemini "
                    "to report by image index, privacy-relevant findings, "
                    "uncertainty, and recommended next tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Local image paths from the post's media_files.",
                        },
                        "question": {
                            "type": "string",
                            "description": (
                                "Specific question for the visual specialist. "
                                "Ask it to inspect images by index and return "
                                "privacy-relevant observations, uncertainty, and "
                                "recommended next actions."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional context accumulated by the Investigator, "
                                "including Qwen/OCR summaries and outputs from prior "
                                "tools such as search, map, OCR, crop, or zoom. "
                                "Use this to let Gemini continue reasoning after "
                                "new evidence arrives."
                            ),
                        },
                    },
                    "required": ["image_paths", "question"],
                },
            },
        }
    raise ValueError(f"unknown delegation tool: {name!r}")


# ── Tool dispatcher: maps name → callable; emits OpenAI schemas ────

class ToolDispatcher:
    """Maps tool names to runnables; emits OpenAI tool schemas.

    Accepts both LangChain BaseTools and plain Python functions. Delegation
    tools are passed in by the caller as a `{name: callable}` dict — they
    are NOT bundled here, because their backend (Gemini-Vertex) is unrelated
    to this module.
    """

    def __init__(
        self,
        tools: list,
        *,
        delegation_tools: dict[str, Callable] | None = None,
        max_output_chars: int = 4000,
    ):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self._exec_tools: dict[str, Any] = {}
        self._schemas: list[dict] = []
        self._max_output = max_output_chars

        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if not name:
                continue
            try:
                schema = convert_to_openai_tool(t)
                self._exec_tools[name] = t
                self._schemas.append(schema)
            except Exception as e:
                logger.warning("ToolDispatcher: skip tool %r: %s", name, e)

        self._delegation = delegation_tools or {}
        for name in self._delegation:
            self._schemas.append(delegation_schema(name))

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    def execute(self, name: str, args: dict) -> str:
        try:
            if name in self._exec_tools:
                tool = self._exec_tools[name]
                if hasattr(tool, "invoke"):
                    out = tool.invoke(args)
                else:
                    out = tool(**args)
            elif name in self._delegation:
                out = self._delegation[name](**args)
            else:
                return f"ERROR: unknown tool {name!r}"
        except Exception as e:
            logger.exception("Tool %r failed: %s", name, e)
            return f"ERROR: {e}"
        if not isinstance(out, str):
            try:
                out = json.dumps(out, ensure_ascii=False, default=str)
            except Exception:
                out = str(out)
        return out[: self._max_output]


# ── Investigator runner (backbone + ReAct + tools) ─────────────────

def make_investigator_runner_openrouter(
    *,
    model: str | None = None,
    tools: list | None = None,
    delegation_tools: dict[str, Callable] | None = None,
    max_iterations: int = 6,
    reasoning_enabled: bool = True,
):
    """Build an Investigator `agent_runner: (prompt) -> (text, tool_call_count)`.

    Args:
      model            : Router model id (default: gpt-5.5).
      tools            : execution tools (defaults to argus.tools.get_all_execution_tools()).
      delegation_tools : optional {name: callable} dict; e.g. provided by
                         argus.adapters.make_delegation_tools_gemini() so the
                         Investigator can offload visual deep-dive to Gemini.
      max_iterations   : max ReAct loop iterations.
      reasoning_enabled: provider-specific reasoning on/off if supported.
    """
    model = model or INVESTIGATOR_MODEL
    if tools is None:
        from argus.tools import get_all_execution_tools
        tools = get_all_execution_tools()

    dispatcher = ToolDispatcher(tools, delegation_tools=delegation_tools)
    client = make_client()

    def runner(prompt: str) -> tuple[str, int]:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_call_count = 0
        last_text = ""

        for iteration in range(1, max_iterations + 1):
            started = time.time()
            argus_log(
                f"llm start provider={PROVIDER_NAME} role=investigator "
                f"model={model} iter={iteration}/{max_iterations}"
            )
            try:
                label = f"provider={PROVIDER_NAME} role=investigator model={model}"
                resp = retry_call(
                    label,
                    lambda: _create_valid_chat_completion(
                        client,
                        label=label,
                        model=model,
                        messages=messages,
                        tools=dispatcher.schemas,
                        tool_choice="auto",
                        extra_body=reasoning_extra_body(reasoning_enabled),
                    ),
                )
            except Exception as e:
                record_llm_call(LLMCallMetric(
                    post_id=current_post_id(),
                    role="investigator",
                    provider=PROVIDER_NAME,
                    model=model,
                    elapsed_seconds=time.time() - started,
                    ok=False,
                    error=str(e),
                ))
                logger.error("Investigator router call failed: %s", e)
                argus_log(
                    f"llm FAILED provider={PROVIDER_NAME} role=investigator "
                    f"iter={iteration} elapsed={time.time() - started:.1f}s error={e}"
                )
                raise

            usage = usage_from_response(resp)
            elapsed = time.time() - started
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="investigator",
                provider=PROVIDER_NAME,
                model=model,
                elapsed_seconds=time.time() - started,
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
                metadata={"tool_call_count": len(resp.choices[0].message.tool_calls or [])},
            ))

            msg = resp.choices[0].message
            content = msg.content or ""
            tool_calls = msg.tool_calls or []
            argus_log(
                f"llm done provider={PROVIDER_NAME} role=investigator "
                f"iter={iteration} elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)} "
                f"tool_calls={len(tool_calls)}"
            )
            if content:
                last_text = content

            messages.append(assistant_message_from_openrouter(msg))

            if not tool_calls:
                return content or last_text, tool_call_count

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                argus_log(f"tool start name={name}")
                tool_started = time.time()
                tool_output = dispatcher.execute(name, args)
                argus_log(
                    f"tool done name={name} elapsed={time.time() - tool_started:.1f}s "
                    f"chars={len(tool_output)}"
                )
                tool_call_count += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

        return last_text, tool_call_count

    return runner


# ── Routing-Verifier runner (backbone text-only) ───────────────────

def make_verifier_runner_openrouter(
    *,
    model: str | None = None,
    reasoning_enabled: bool = True,
):
    """Build a text-only `(prompt) -> str` runner for Routing-Verifier."""
    model = model or VERIFIER_MODEL
    client = make_client()

    def runner(prompt: str) -> str:
        started = time.time()
        argus_log(f"llm start provider={PROVIDER_NAME} role=routing_verifier model={model}")
        try:
            label = f"provider={PROVIDER_NAME} role=routing_verifier model={model}"
            resp = retry_call(
                label,
                lambda: _create_valid_chat_completion(
                    client,
                    label=label,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=reasoning_extra_body(reasoning_enabled),
                ),
            )
            usage = usage_from_response(resp)
            elapsed = time.time() - started
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="routing_verifier",
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
            ))
            argus_log(
                f"llm done provider={PROVIDER_NAME} role=routing_verifier "
                f"elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)}"
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="routing_verifier",
                provider=PROVIDER_NAME,
                model=model,
                elapsed_seconds=time.time() - started,
                ok=False,
                error=str(e),
            ))
            logger.error("Routing verifier router call failed: %s", e)
            argus_log(
                f"llm FAILED provider={PROVIDER_NAME} role=routing_verifier "
                f"elapsed={time.time() - started:.1f}s error={e}"
            )
            raise

    return runner


# ── Narrative runner (profile Layer 2) ─────────────────────────────

def make_narrative_runner_openrouter(
    *,
    model: str | None = None,
    reasoning_enabled: bool = True,
):
    """Build a `(prompt) -> str` runner for profile narrative synthesis."""
    model = model or NARRATIVE_MODEL
    client = make_client()

    def runner(prompt: str) -> str:
        started = time.time()
        argus_log(f"llm start provider={PROVIDER_NAME} role=narrative model={model}")
        try:
            label = f"provider={PROVIDER_NAME} role=narrative model={model}"
            resp = retry_call(
                label,
                lambda: _create_valid_chat_completion(
                    client,
                    label=label,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=reasoning_extra_body(reasoning_enabled),
                ),
            )
            usage = usage_from_response(resp)
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="narrative",
                provider=PROVIDER_NAME,
                model=model,
                elapsed_seconds=time.time() - started,
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
                f"llm done provider={PROVIDER_NAME} role=narrative "
                f"elapsed={time.time() - started:.1f}s tokens={usage.get('total_tokens', 0)}"
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="narrative",
                provider=PROVIDER_NAME,
                model=model,
                elapsed_seconds=time.time() - started,
                ok=False,
                error=str(e),
            ))
            logger.error("Narrative router call failed: %s", e)
            argus_log(
                f"llm FAILED provider={PROVIDER_NAME} role=narrative "
                f"elapsed={time.time() - started:.1f}s error={e}"
            )
            raise

    return runner
