"""Adapters — bridge ARGUS components to native LLM backends.

Backend assignments (revision §3.6 + your env config):

    Investigator (backbone)  -> argus/openrouter.py     (OpenAI-compatible router)
    Verifier    (backbone)   -> Qwen/GPT, configured by ARGUS_VERIFIER_PROVIDER
    Narrative   (backbone)   -> argus/openrouter.py     (OpenAI-compatible router)
    ─── this file ────────────────────────────────────────────────────
    Perception  (Qwen3.6-Plus) -> DashScope OpenAI-compat (DASHSCOPE_API_KEY)
    Verifier    (Qwen3.7-Max)  -> DashScope OpenAI-compat (DASHSCOPE_API_KEY)
    AMTR fallb. (Qwen3.7-Max)  -> DashScope OpenAI-compat (DASHSCOPE_API_KEY)
    Visual specialist (Gemini) -> Vertex AI native       (GOOGLE_KEY_PATH)
    OCR (PaddleOCR)            -> local

Each builder returns a small callable matching the interface ARGUS components
expect. They are intentionally thin so a stub can replace any of them in
tests / ablation runs.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable

from argus.llm.gemini import (
    DELEGATION_VISUAL_MODEL,
    generate_vision as gemini_vision,
)
from argus.llm.qwen import (
    AMTR_FALLBACK_MODEL,
    PERCEPTION_VL_MODEL,
    chat_text as qwen_text,
    chat_vision as qwen_vision,
)

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_PROVIDER = "qwen"
DEFAULT_QWEN_VERIFIER_MODEL = "qwen3.7-max"


# ════════════════════════════════════════════════════════════════════
# Perception VL — Qwen3.6-Plus via DashScope (OpenAI-compatible)
# ════════════════════════════════════════════════════════════════════

def make_vl_fn_qwen_dashscope(*, model: str | None = None):
    """Build a `vl_fn(image_paths, prompt) -> str` using Qwen3.6-Plus via DashScope."""
    model_name = model or PERCEPTION_VL_MODEL

    def vl(image_paths: list[str], prompt: str) -> str:
        if not image_paths:
            return ""
        return qwen_vision(
            image_paths,
            prompt,
            model=model_name,
            role="perception_vl",
        )

    return vl


# ════════════════════════════════════════════════════════════════════
# AMTR LLM fallback — Qwen via DashScope (text only)
# ════════════════════════════════════════════════════════════════════

_AMTR_FALLBACK_SYSTEM = (
    "You are a routing decision module for the ARGUS privacy investigation "
    "agent. Given a perceptual signature and a target hypothesis, choose the "
    "best (model, tool_family) for the next investigation step.\n\n"
    "Allowed models       : gemini, qwen, claude, gpt, auto\n"
    "Allowed tool_families: map_search, web_search, ocr, zoom, "
    "fetch, stop\n\n"
    "Return strict JSON only: "
    '{"model": "<one of above>", "tool_family": "<one of above>", '
    '"rationale": "<one short sentence>"}'
)


def make_amtr_fallback_qwen_dashscope(*, model: str | None = None):
    """Build an AMTR LLM-fallback `(state_dict) -> {model, tool_family, rationale}`."""
    model_name = model or AMTR_FALLBACK_MODEL

    def runner(state: dict) -> dict:
        sig = state.get("perceptual_signature")
        sig_view = {
            "vl_tag":     getattr(sig, "vl_tag", "") if sig else "",
            "vl_caption": getattr(sig, "vl_caption", "") if sig else "",
            "image_summaries": getattr(sig, "image_summaries", []) if sig else [],
            "entities":   dict(getattr(sig, "entities", {})) if sig else {},
        }
        prompt = (
            f"Attribute       : {state.get('attribute') or '(none)'}\n"
            f"Attribute class : {state.get('attribute_class') or '(none)'}\n"
            f"Region hint     : {state.get('region', 'unknown')}\n"
            f"Active attrs    : {state.get('active_attributes', [])}\n"
            f"Signature       : {json.dumps(sig_view, ensure_ascii=False)}\n"
        )
        try:
            text = qwen_text(
                [
                    {"role": "system", "content": _AMTR_FALLBACK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                model=model_name,
                role="amtr_fallback",
            )
            j_start = text.find("{")
            j_end = text.rfind("}") + 1
            if j_start >= 0 and j_end > j_start:
                return json.loads(text[j_start:j_end])
        except Exception as e:
            logger.warning("AMTR fallback failed: %s", e)
        return {}

    return runner


# ════════════════════════════════════════════════════════════════════
# Routing verifier — text-only, independent context
# ════════════════════════════════════════════════════════════════════

def make_verifier_runner_qwen_dashscope(*, model: str | None = None):
    """Build a Routing-Verifier `(prompt) -> str` using Qwen3.7-Max via DashScope."""
    model_name = (
        model
        or os.environ.get("ARGUS_VERIFIER_MODEL")
        or DEFAULT_QWEN_VERIFIER_MODEL
    )

    def runner(prompt: str) -> str:
        return qwen_text(
            [{"role": "user", "content": prompt}],
            model=model_name,
            role="routing_verifier",
            enable_thinking=True,
        )

    return runner


def make_verifier_runner(*, provider: str | None = None, model: str | None = None):
    """Build the configured Routing-Verifier runner.

    Defaults to Qwen because the verifier is a text-only evidence/reasoning
    judge. GPT/OpenAI-compatible routing remains available for ablations by
    setting ARGUS_VERIFIER_PROVIDER=gpt and ARGUS_VERIFIER_MODEL accordingly.
    """
    provider_name = (
        provider
        or os.environ.get("ARGUS_VERIFIER_PROVIDER")
        or DEFAULT_VERIFIER_PROVIDER
    ).strip().lower()

    if provider_name in {"qwen", "dashscope"}:
        return make_verifier_runner_qwen_dashscope(model=model)

    if provider_name in {"gpt", "openrouter", "router", "shadowapi"}:
        from argus.openrouter import make_verifier_runner_openrouter

        return make_verifier_runner_openrouter(model=model)

    raise ValueError(
        "Unsupported ARGUS_VERIFIER_PROVIDER="
        f"{provider_name!r}; expected qwen or gpt/openrouter."
    )


# ════════════════════════════════════════════════════════════════════
# Visual delegation tool — Gemini (Vertex AI native)
# This is passed into the backbone Investigator as `delegation_tools=...`.
# ════════════════════════════════════════════════════════════════════

def make_delegation_tools_gemini() -> dict[str, Callable]:
    """Return Gemini visual specialist tools for the Investigator."""
    def deep_visual_analysis(image_paths: list[str], question: str, context: str = "") -> str:
        if not image_paths:
            return "no images provided"
        full_question = question
        if context.strip():
            full_question = (
                "You are the visual reasoning specialist inside an iterative "
                "privacy-investigation loop. The Investigator may have already "
                "run tools such as web search, map search, OCR, crop, or zoom. "
                "Use the context below together with the original images. Do "
                "not call tools yourself; only reason visually and recommend "
                "what the Investigator should do next.\n\n"
                f"CONTEXT FROM INVESTIGATOR AND PRIOR TOOLS:\n{context}\n\n"
                f"CURRENT QUESTION:\n{question}"
            )
        try:
            return gemini_vision(
                image_paths,
                full_question,
                model=DELEGATION_VISUAL_MODEL,
                role="deep_visual_analysis",
            )
        except Exception as e:
            logger.error("deep_visual_analysis failed: %s", e)
            return f"ERROR: {e}"

    return {
        "deep_visual_analysis":  deep_visual_analysis,
    }


# ════════════════════════════════════════════════════════════════════
# OCR (PaddleOCR, local — no network)
# ════════════════════════════════════════════════════════════════════

def make_ocr_fn():
    """Return an `ocr_fn(image_path) -> str` using ARGUS-local tools."""
    from argus.tools import run_ocr

    def ocr(path: str) -> str:
        try:
            text = run_ocr(path)
            return "" if text.startswith("Error ") else text
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", path, exc)
            return ""

    return ocr


# Helpers live in argus.llm.* so the backend plumbing stays in one place.
