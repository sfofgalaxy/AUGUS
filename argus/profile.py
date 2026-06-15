"""Profile synthesis — two-layer design.

Layer 1 (deterministic):
    `synthesize_profile(cpeg)` → UserProfile.attributes
    Walks active hypotheses, picks best per attribute, emits structured
    fields (value, level, granularity_score, evidence_ids). No LLM. Used
    by EGHE evaluation; fully reproducible.

Layer 2 (LLM, optional):
    `synthesize_narrative(profile, cpeg, runner)` → narrative dict
    Adds cross-attribute insights, implicit attributes, multi-value merges,
    and a natural-language paragraph. Each insight cites
    `derived_from: [attr, ...]` for traceability. Used for case studies and
    human inspection. Skipped if `runner` is None.

Why two layers (revision §3.4 / discussion):
    - EGHE Tier 1 needs deterministic granularity_score; LLM rewriting would
      destroy reproducibility.
    - But the structured projection misses cross-attribute coherence,
      implicit descriptors, and multi-value cases. Layer 2 fills these
      without polluting Layer 1.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from argus.cpeg import CPEG
from argus.types import UserProfile
from argus.attributes import (
    ALL_ATTRIBUTES,
    get_levels,
    get_sensitivity,
    level_score,
)

logger = logging.getLogger(__name__)


# ── Selection policy ──

# A hypothesis is "trusted" if its confidence exceeds this and at least one
# supporting evidence exists. Lower trust → "unverified" annotation.
TRUST_THRESHOLD = 0.55


def synthesize_profile(cpeg: CPEG, *, metadata: dict[str, Any] | None = None) -> UserProfile:
    """Project CPEG into a final UserProfile."""
    profile = UserProfile(user_id=cpeg.user_id, metadata=metadata or {})

    attrs = sorted(set(ALL_ATTRIBUTES) | {h.attribute for h in cpeg.hypotheses.values()})
    for attr in attrs:
        hyps = cpeg.hypotheses_for(attr, only_active=True)
        if not hyps:
            continue
        best = _select_best(cpeg, hyps)
        if best is None:
            continue
        h, support_score, sup_evs = best
        levels = get_levels(attr)
        level_idx = min(h.level, len(levels) - 1)
        is_known_attribute = attr in ALL_ATTRIBUTES
        profile.attributes[attr] = {
            "value": h.value,
            "level_idx": level_idx,
            "level_name": levels[level_idx],
            "confidence": round(h.confidence, 3),
            "trusted": h.confidence >= TRUST_THRESHOLD and support_score > 0,
            "support_score": round(support_score, 3),
            "evidence_ids": sup_evs,
            "sensitivity": get_sensitivity(attr),
            "granularity_score": level_score(attr, level_idx),
            "hypothesis_id": h.id,
            "schema_known": is_known_attribute,
        }

    profile.metadata.update(_aggregate_metadata(cpeg))
    return profile


# ── Helpers ──

def _select_best(cpeg, hyps):
    """Pick the (hypothesis, support_score, supporting_ev_ids) with the best
    composite score. Composite = confidence * (1 + support_score) ^ 0.5,
    breaking ties by deepest level then most evidence."""
    scored = []
    for h in hyps:
        s = cpeg.support_score(h.id)
        sup, _ = cpeg.evidence_for_hypothesis(h.id)
        composite = h.confidence * (1.0 + s) ** 0.5
        scored.append((composite, h.level, len(sup), h, s, [e.id for e in sup]))

    if not scored:
        return None
    scored.sort(reverse=True, key=lambda t: (t[0], t[1], t[2]))
    composite, level, _n, h, s, sup_ids = scored[0]
    return h, s, sup_ids


def _aggregate_metadata(cpeg) -> dict[str, Any]:
    n_hyp_total = len(cpeg.hypotheses)
    n_hyp_active = sum(1 for h in cpeg.hypotheses.values() if h.is_active)
    n_evidence = len(cpeg.evidence)
    n_chains = len(cpeg.chains)

    if cpeg.hypotheses:
        confidences = [h.confidence for h in cpeg.hypotheses.values() if h.is_active]
        mean_conf = sum(confidences) / max(1, len(confidences))
    else:
        mean_conf = 0.0

    return {
        "n_posts": len(cpeg.posts),
        "n_hypotheses_total": n_hyp_total,
        "n_hypotheses_active": n_hyp_active,
        "n_evidence": n_evidence,
        "n_chain_edges": n_chains,
        "mean_active_confidence": round(mean_conf, 3),
    }


# ── Layer 2: LLM-synthesized narrative ─────────────────────────────

_NARRATIVE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "narrative.txt"


def synthesize_narrative(
    profile: UserProfile,
    cpeg: CPEG,
    *,
    runner: Callable[[str], str] | None,
) -> dict[str, Any]:
    """Layer 2: LLM narrative on top of the structured profile.

    Returns the parsed narrative dict (cross_attribute_insights,
    implicit_attributes, merged_multi_value, narrative). Returns {} on
    runner=None or any failure (Layer 1 stays valid regardless).
    """
    if runner is None:
        return {}
    if not profile.attributes:
        return {}                                            # nothing to narrate
    try:
        system_prompt = _NARRATIVE_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as e:
        logger.error("Could not read narrative prompt: %s", e)
        return {}

    prompt = _build_narrative_prompt(system_prompt, profile, cpeg)
    raw = runner(prompt) or ""
    return _parse_narrative_json(raw)


def synthesize_full(
    cpeg: CPEG,
    *,
    metadata: dict[str, Any] | None = None,
    narrative_runner: Callable[[str], str] | None = None,
) -> UserProfile:
    """Convenience: produce both Layer 1 and (optionally) Layer 2 in one call."""
    profile = synthesize_profile(cpeg, metadata=metadata)
    if narrative_runner is not None:
        profile.narrative = synthesize_narrative(profile, cpeg, runner=narrative_runner)
    return profile


def _build_narrative_prompt(system: str, profile: UserProfile, cpeg: CPEG) -> str:
    parts: list[str] = [system, "\n\n## STRUCTURED PROFILE (Layer 1, deterministic)\n"]

    if not profile.attributes:
        parts.append("(empty)\n")
    else:
        for attr, info in sorted(profile.attributes.items()):
            parts.append(
                f"- {attr}\n"
                f"    value     : {info.get('value')}\n"
                f"    level     : {info.get('level_name')} (idx={info.get('level_idx')})\n"
                f"    granularity_score : {info.get('granularity_score')}\n"
                f"    confidence: {info.get('confidence')}\n"
                f"    sensitivity: {info.get('sensitivity')}\n"
                f"    trusted   : {info.get('trusted')}"
            )

    parts.append("\n## CPEG SUMMARY")
    parts.append(f"- posts         : {len(cpeg.posts)}")
    parts.append(f"- evidence      : {len(cpeg.evidence)}")
    parts.append(f"- hypotheses    : {len(cpeg.hypotheses)}")
    parts.append(f"- chain_edges   : {len(cpeg.chains)}")

    if cpeg.chains:
        parts.append("\n  Chain rules invoked:")
        for c in cpeg.chains[:8]:
            parts.append(f"    - {c.rule}")

    # Top-supported evidence (up to 6) for grounding the narrative
    top_ev = sorted(
        cpeg.evidence.values(),
        key=lambda e: e.confidence,
        reverse=True,
    )[:6]
    if top_ev:
        parts.append("\n## TOP EVIDENCE (for grounding insights)")
        for e in top_ev:
            parts.append(f"- [{e.id}] ({e.source_tool}, conf={e.confidence:.2f}) {e.text[:160]}")

    parts.append("\n## YOUR TASK")
    parts.append(
        "Produce the narrative JSON per the system prompt. Cite "
        "`derived_from: [attribute_slot, ...]` for every insight and "
        "implicit_attribute. Do not contradict Layer 1."
    )
    return "\n".join(parts)


def _parse_narrative_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    j_start = raw.find("{")
    j_end = raw.rfind("}") + 1
    if j_start < 0 or j_end <= j_start:
        return {}
    try:
        return json.loads(raw[j_start:j_end])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Narrative JSON parse failed: %s", e)
        return {}
