"""Investigator — one ReAct step.

Per call:
  1. Build a focused user prompt from (post + perceptual_signature + active CPEG
     evidence + target hypothesis + chosen tool family).
  2. Invoke the LLM agent (which may call tools via function-calling).
  3. Parse the JSON output into a list of (Evidence, finding-attrs) records.

The Investigator does NOT loop — the outer pipeline drives the loop. This
keeps the component simple and makes ablation possible (one step at a time).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from argus.types import (
    PerceptualSignature,
    Hypothesis,
    Evidence,
    RouteDecision,
)

logger = logging.getLogger(__name__)


# ── Investigator output (returned to caller) ──

@dataclass
class StepResult:
    """One Investigator-step result, fed to RoutingVerifier."""
    raw_output: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    suggested_action: str = "CONTINUE"
    rationale: str = ""
    tool_call_count: int = 0


# ── Investigator ──

class Investigator:
    """Executes one investigation step.

    Args:
      agent_runner : callable (prompt: str) -> tuple[str, int]
                     — invokes the underlying deep-agent / tool-calling LLM
                     and returns (output_text, tool_call_count).
      prompt_path  : path to the system prompt template (defaults to
                     argus/prompts/investigator.txt).
    """

    def __init__(
        self,
        *,
        agent_runner,
        prompt_path: Path | None = None,
    ):
        self.agent_runner = agent_runner
        self.system_prompt = (prompt_path or _default_prompt_path("investigator.txt")).read_text(encoding="utf-8")

    # ── Public API ──

    def run_step(
        self,
        *,
        post: dict[str, Any],
        signature: PerceptualSignature,
        target_hypothesis: Hypothesis | None,
        route: RouteDecision,
        prior_evidence: list[Evidence],
    ) -> StepResult:
        """Run one investigation step and return a StepResult."""
        prompt = self._build_user_prompt(
            post=post,
            signature=signature,
            target_hypothesis=target_hypothesis,
            route=route,
            prior_evidence=prior_evidence,
        )

        output, tool_call_count = self._invoke(prompt)
        findings, suggested, rationale = self._parse_output(output)

        return StepResult(
            raw_output=output,
            findings=findings,
            suggested_action=suggested,
            rationale=rationale,
            tool_call_count=tool_call_count,
        )

    # ── Wiring ──

    def _invoke(self, user_prompt: str) -> tuple[str, int]:
        result = self.agent_runner(self._wrap_prompt(user_prompt))
        if isinstance(result, tuple) and len(result) == 2:
            count = result[1]
            if isinstance(count, int):
                return str(result[0] or ""), count
            if isinstance(count, list):
                return str(result[0] or ""), len(count)
            return str(result[0] or ""), 0
        return str(result or ""), 0

    def _wrap_prompt(self, user_prompt: str) -> str:
        return f"{self.system_prompt}\n\n---\n\n{user_prompt}"

    # ── Prompt construction ──

    def _build_user_prompt(
        self,
        *,
        post: dict[str, Any],
        signature: PerceptualSignature,
        target_hypothesis: Hypothesis | None,
        route: RouteDecision,
        prior_evidence: list[Evidence],
    ) -> str:
        parts: list[str] = []

        # Post identity
        parts.append(f"## Post ID: {post.get('post_id', signature.post_id)}")
        if signature.raw_post_text:
            parts.append(f"\n## Caption\n{signature.raw_post_text}")
        if post.get("timestamp"):
            parts.append(f"\nTimestamp: {post['timestamp']}")
        if post.get("location_ip"):
            parts.append(f"Recent IP location: {post['location_ip']}")

        # Media
        media = post.get("media_files") or []
        if media:
            parts.append(f"\n## Media ({len(media)} files)")
            for i, m in enumerate(media, start=1):
                parts.append(f"- Image {i}: {m}")

        # Perceptual signature (compact)
        parts.append("\n## Perceptual Signature")
        parts.append(f"- Post primary VL tag: {signature.vl_tag or '(none)'}")
        if signature.image_summaries:
            parts.append("- Per-image VL:")
            for item in signature.image_summaries:
                idx = item.get("image_index", "?")
                tag = item.get("vl_tag") or "none"
                caption = item.get("vl_caption") or "(none)"
                parts.append(f"    Image {idx}: tag={tag}; caption={caption}")
                if item.get("ocr_text"):
                    ocr_item = str(item["ocr_text"])
                    parts.append(f"      OCR: {ocr_item[:240]}")
                if item.get("entities"):
                    parts.append(f"      Entities: {', '.join(map(str, item['entities']))}")
        else:
            parts.append(f"- VL caption: {signature.vl_caption or '(none)'}")
        if signature.ocr_text:
            ocr_preview = signature.ocr_text[:600] + ("…" if len(signature.ocr_text) > 600 else "")
            parts.append(f"- OCR text  : {ocr_preview}")
        if signature.entities:
            parts.append("- Entities  :")
            for k, vs in signature.entities.items():
                parts.append(f"    {k}: {', '.join(map(str, vs))}")

        parts.append("\n## Attribute Slots")
        parts.append(
            "Use a concise dotted attribute name for every finding.attribute. "
            "Prefer existing ARGUS-style names when they fit, but do not discard "
            "a privacy-relevant finding just because it does not fit a fixed schema. "
            "Additional fields such as interest.hobby are allowed and should be preserved."
        )

        # Target hypothesis (if any)
        parts.append("\n## Target Hypothesis (HDI)")
        if target_hypothesis is not None:
            parts.append(
                f"- Attribute : {target_hypothesis.attribute}\n"
                f"- Value     : {target_hypothesis.value}\n"
                f"- Level     : {target_hypothesis.level} (max {target_hypothesis.max_level})\n"
                f"- Confidence: {target_hypothesis.confidence:.2f}\n"
                f"- Status    : {target_hypothesis.status}"
            )
        else:
            parts.append("- (no active hypothesis; this is a discovery step)")

        # Route decision
        parts.append("\n## Routing Decision (AMTR)")
        parts.append(
            f"- Use tool family : `{route.tool_family}`\n"
            f"- Suggested model : `{route.model}`\n"
            f"- Rationale       : {route.rationale}"
        )

        # Prior evidence (CPEG slice for this attribute)
        if prior_evidence:
            parts.append(f"\n## Prior Evidence ({len(prior_evidence)} items)")
            for e in prior_evidence[:8]:
                snippet = (e.text or "")[:200]
                parts.append(f"- [{e.source_tool}] ({e.confidence:.2f}) {snippet}")

        # Instructions
        parts.append(
            "\n## Instructions\n"
            f"AMTR suggests `{route.tool_family}`, but this is advisory. "
            "You are the manager: first decide whether the Qwen/OCR signature "
            "already supports privacy findings, whether Gemini should inspect "
            "the original images via `deep_visual_analysis`, or whether a search/map/OCR/zoom "
            "tool is more appropriate. If you call `deep_visual_analysis`, ask it "
            "to inspect images by index and report: current image number, visible "
            "privacy-relevant details, uncertainty, and suggested next tool. "
            "If you then call an execution tool, call `deep_visual_analysis` again "
            "with a concise `context` containing the relevant tool output so Gemini "
            "can continue reasoning over the images plus the new evidence. Gemini "
            "does not call tools; you decide and execute tools. Make only the tool "
            "calls needed for this decision, then return strict JSON per the system prompt."
        )
        return "\n".join(parts)

    # ── Output parsing ──

    @staticmethod
    def _parse_output(text: str) -> tuple[list[dict[str, Any]], str, str]:
        if not text:
            return [], "STOP", "empty output"
        # Find first balanced JSON object
        try:
            j_start = text.find("{")
            j_end = text.rfind("}") + 1
            if j_start < 0 or j_end <= j_start:
                return [], "CONTINUE", "no JSON detected"
            data = json.loads(text[j_start:j_end])
        except (json.JSONDecodeError, ValueError) as exc:
            return [], "CONTINUE", f"json parse error: {exc}"

        findings = data.get("findings") or []
        if not isinstance(findings, list):
            findings = []

        suggested = str(data.get("next_action_suggestion") or "CONTINUE").upper()
        if suggested not in {"CONTINUE", "ESCALATE", "REFUTE", "BRANCH", "STOP"}:
            suggested = "CONTINUE"

        rationale = str(data.get("rationale") or "")
        return findings, suggested, rationale


def _default_prompt_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / filename
