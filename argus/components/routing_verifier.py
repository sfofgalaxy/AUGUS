"""Routing-Verifier — claim verification + HDI action proposal + AMTR routing.

This is the brain of the loop. After Investigator runs one step, the
Routing-Verifier:

  1. Verifies the Investigator's findings against ground truth (perception
     OCR + tool receipts + prior CPEG evidence).
  2. Proposes a confidence delta for each touched hypothesis.
  3. Picks one HDI action (CONTINUE / ESCALATE / REFUTE / BRANCH / STOP).
  4. If CONTINUE/ESCALATE/BRANCH, hands the next route over to AMTR.

The verifier should run in an independent context from the Investigator. In
the default ARGUS stack it uses Qwen3.7-Max, while the Investigator backbone
uses GPT and Gemini remains the visual specialist.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from argus.types import (
    Hypothesis,
    HDIAction,
    PerceptualSignature,
    RouteDecision,
    Evidence,
)
from argus.hypothesis import HypothesisPool, reconcile_action
from argus.routing.amtr import AMTR
from argus.cpeg import CPEG
from argus.logging_utils import argus_log

logger = logging.getLogger(__name__)


# ── Verifier output ──

@dataclass
class VerificationResult:
    action: HDIAction
    target_hypothesis_id: str | None
    next_route: RouteDecision | None        # None iff action == STOP
    hypothesis_updates: list[dict[str, Any]] = field(default_factory=list)
    claim_verifications: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_score: float = 0.5
    rationale: str = ""
    raw_output: str = ""
    new_evidence: list[Evidence] = field(default_factory=list)


# ── RoutingVerifier ──

class RoutingVerifier:
    """One LLM call that decides: verify, update, route.

    Args:
      llm_runner : callable (prompt: str) -> str (text completion).
      amtr       : AMTR instance.
      prompt_path: optional system-prompt path.
    """

    def __init__(
        self,
        *,
        llm_runner,
        amtr: AMTR,
        prompt_path: Path | None = None,
    ):
        self.llm_runner = llm_runner
        self.amtr = amtr
        self.system_prompt = (prompt_path or _default_prompt_path("routing_verifier.txt")).read_text(encoding="utf-8")

    # ── Public API ──

    def verify_and_route(
        self,
        *,
        signature: PerceptualSignature,
        target_hypothesis: Hypothesis | None,
        investigator_output: str,
        investigator_findings: list[dict[str, Any]],
        prior_evidence: list[Evidence],
        pool: HypothesisPool,
        cpeg: CPEG,
        post_id: str,
        step_id: int,
    ) -> VerificationResult:
        # 1. LLM call
        prompt = self._build_prompt(
            signature=signature,
            target_hypothesis=target_hypothesis,
            investigator_output=investigator_output,
            investigator_findings=investigator_findings,
            prior_evidence=prior_evidence,
            pool=pool,
        )
        raw = self._invoke(prompt)
        parsed = self._parse(raw)
        status_counts = _claim_status_counts(parsed.get("claim_verifications") or [])
        argus_log(
            "verifier claim_status "
            f"supported={status_counts['SUPPORTED']} "
            f"unsupported={status_counts['UNSUPPORTED']} "
            f"contradicted={status_counts['CONTRADICTED']} "
            f"unknown={status_counts['UNKNOWN']}"
        )

        # 2. Apply hypothesis updates (deltas) to pool
        updates: list[dict[str, Any]] = []
        for u in parsed.get("hypothesis_updates", []):
            hid = u.get("hypothesis_id")
            delta = float(u.get("confidence_delta", 0.0))
            reason = str(u.get("reason", ""))
            if hid in pool._by_id:                   # skip unknown
                pool.update_confidence(hid, delta, step=step_id, reason=reason)
                updates.append({"hypothesis_id": hid, "delta": delta, "reason": reason})

        # 3. Convert findings → Evidence and link to CPEG
        new_evidence = self._materialize_findings(
            findings=investigator_findings,
            pool=pool,
            cpeg=cpeg,
            post_id=post_id,
            target_hypothesis=target_hypothesis,
            verifier_claims=parsed.get("claim_verifications") or [],
        )

        # 4. Apply chaining rules (privacy-specific cross-attribute inference)
        cpeg.apply_chaining_rules()

        # 5. Reconcile proposed action against pool state.
        proposed_str = parsed.get("action", "CONTINUE").upper()
        try:
            proposed = HDIAction[proposed_str]
        except KeyError:
            proposed = HDIAction.CONTINUE
        proposed = _soften_overstrict_action(proposed, parsed)
        reconciled = reconcile_action(
            proposed,
            target_hypothesis=target_hypothesis,
            pool=pool,
        )

        # 6. Apply ESCALATE / BRANCH side effects
        new_target_hypothesis = target_hypothesis
        action_args = parsed.get("action_args") or {}
        if reconciled == HDIAction.ESCALATE and target_hypothesis is not None:
            new_value = action_args.get("new_value") or target_hypothesis.value
            new_h = pool.escalate(
                target_hypothesis.id, new_value=new_value, step=step_id,
            )
            if new_h is not None:
                cpeg.add_hypothesis(new_h)
                new_target_hypothesis = new_h
        elif reconciled == HDIAction.BRANCH and target_hypothesis is not None:
            branch_value = action_args.get("branch_value") or action_args.get("new_value")
            if branch_value:
                sib = pool.branch(target_hypothesis.id, new_value=branch_value, step=step_id)
                cpeg.add_hypothesis(sib)
        elif reconciled == HDIAction.REFUTE and target_hypothesis is not None:
            pool.refute(target_hypothesis.id, step=step_id, reason=parsed.get("rationale", ""))

        # 7. Compute next route (or None on STOP)
        next_route: RouteDecision | None = None
        if reconciled in (HDIAction.CONTINUE, HDIAction.ESCALATE, HDIAction.BRANCH):
            target = new_target_hypothesis if new_target_hypothesis is not None else target_hypothesis
            next_route = self.amtr.decide(
                signature=signature,
                hypothesis=target,
                active_attributes=pool.attributes_with_active(),
            )

        return VerificationResult(
            action=reconciled,
            target_hypothesis_id=(
                new_target_hypothesis.id if new_target_hypothesis is not None else None
            ),
            next_route=next_route,
            hypothesis_updates=updates,
            claim_verifications=parsed.get("claim_verifications", []),
            uncertainty_score=float(parsed.get("uncertainty_score", 0.5)),
            rationale=str(parsed.get("rationale", "")),
            raw_output=raw,
            new_evidence=new_evidence,
        )

    # ── Helpers ──

    def _materialize_findings(
        self,
        *,
        findings: list[dict[str, Any]],
        pool: HypothesisPool,
        cpeg: CPEG,
        post_id: str,
        target_hypothesis: Hypothesis | None,
        verifier_claims: list[dict[str, Any]],
    ) -> list[Evidence]:
        """Turn Investigator findings into CPEG Evidence + link to hypotheses.

        For each finding:
          - Create a new Hypothesis or attach to an existing one with same
            (attribute, value).
          - Create one Evidence node for the finding's evidence_chain.
          - Link support / refute based on verifier status.
        """
        new_ev: list[Evidence] = []
        for i, f in enumerate(findings):
            attr = _normalize_attribute(f.get("attribute") or (target_hypothesis.attribute if target_hypothesis else None))
            value = f.get("value")
            level = int(f.get("level", 0))
            confidence = float(f.get("confidence", 0.5))
            source_tool = str(f.get("source_tool") or "unknown")
            evidence_chain = str(f.get("evidence_chain") or "")
            if not attr or not value:
                continue
            status = _status_for_finding(i, evidence_chain, verifier_claims)
            if status == "UNSUPPORTED":
                # Weak/speculative findings should not enter CPEG/profile at all.
                # The raw investigator/verifier outputs remain in step_logs.
                continue
            weight = {"SUPPORTED": 1.0, "CONTRADICTED": 0.7}.get(status, 1.0)

            # Find or create hypothesis
            existing = [h for h in pool.active_for(attr) if h.value.strip().lower() == value.strip().lower()]
            if existing:
                h = existing[0]
            else:
                h = pool.add(attribute=attr, value=value, level=level, confidence=confidence)
                cpeg.add_hypothesis(h)

            # Create evidence
            ev_id = f"{post_id}-{step_id_safe(i)}"
            e = Evidence(
                id=ev_id,
                text=evidence_chain or f"{attr}={value} (level {level}) via {source_tool}",
                source_tool=source_tool,
                modality="tool_result" if source_tool != "ocr" else "ocr",
                confidence=confidence,
                post_id=post_id,
                raw_data={"finding": f, "verifier_status": status},
            )
            cpeg.add_evidence(e)
            new_ev.append(e)

            if status == "CONTRADICTED":
                cpeg.link_refute(e.id, h.id, weight=weight, note=status)
            else:
                cpeg.link_support(e.id, h.id, weight=weight, note=status)
        return new_ev

    # ── Prompt + LLM ──

    def _build_prompt(
        self,
        *,
        signature: PerceptualSignature,
        target_hypothesis: Hypothesis | None,
        investigator_output: str,
        investigator_findings: list[dict[str, Any]],
        prior_evidence: list[Evidence],
        pool: HypothesisPool,
    ) -> str:
        parts: list[str] = []
        parts.append("## Perceptual Signature")
        parts.append(f"- Post primary VL tag: {signature.vl_tag or '(none)'}")
        if signature.image_summaries:
            parts.append("- Per-image VL:")
            for item in signature.image_summaries:
                idx = item.get("image_index", "?")
                tag = item.get("vl_tag") or "none"
                caption = item.get("vl_caption") or "(none)"
                parts.append(f"    Image {idx}: tag={tag}; caption={caption}")
        else:
            parts.append(f"- VL caption: {signature.vl_caption or '(none)'}")
        if signature.ocr_text:
            parts.append(f"- OCR text  : {signature.ocr_text[:500]}")

        parts.append("\n## Target Hypothesis")
        if target_hypothesis is not None:
            parts.append(
                f"- ID         : {target_hypothesis.id}\n"
                f"- Attribute  : {target_hypothesis.attribute}\n"
                f"- Value      : {target_hypothesis.value}\n"
                f"- Level      : {target_hypothesis.level} / {target_hypothesis.max_level}\n"
                f"- Confidence : {target_hypothesis.confidence:.2f}"
            )
        else:
            parts.append("(none — discovery mode)")

        parts.append("\n## Active Hypotheses (excerpt)")
        for h in pool.active()[:10]:
            parts.append(f"- {h.id} | {h.attribute} = {h.value} | "
                         f"level={h.level}/{h.max_level} | conf={h.confidence:.2f}")

        parts.append("\n## Investigator Output (raw)")
        parts.append(investigator_output[:2500])

        parts.append("\n## Investigator Findings (parsed)")
        for f in investigator_findings:
            parts.append(f"- {f}")

        if prior_evidence:
            parts.append("\n## Prior Evidence")
            for e in prior_evidence[:6]:
                parts.append(f"- [{e.source_tool}] ({e.confidence:.2f}) {e.text[:160]}")

        parts.append("\n## Your Task")
        parts.append(
            "Verify each Investigator claim, propose hypothesis updates, and "
            "decide the next HDI action. Output strict JSON per the system prompt."
        )
        return f"{self.system_prompt}\n\n---\n\n" + "\n".join(parts)

    def _invoke(self, prompt: str) -> str:
        return str(self.llm_runner(prompt) or "")

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        if not raw:
            return {"action": "STOP", "uncertainty_score": 1.0}
        try:
            j_start = raw.find("{")
            j_end = raw.rfind("}") + 1
            if j_start < 0 or j_end <= j_start:
                return {"action": "CONTINUE", "uncertainty_score": 0.7}
            return json.loads(raw[j_start:j_end])
        except (json.JSONDecodeError, ValueError):
            return {"action": "CONTINUE", "uncertainty_score": 0.7}


def step_id_safe(i: int) -> str:
    """Stable, short suffix used inside Evidence IDs."""
    import uuid
    return f"{i}-{uuid.uuid4().hex[:8]}"


def _normalize_attribute(attr: Any) -> str | None:
    if attr is None:
        return None
    raw = str(attr).strip()
    if not raw:
        return None
    return raw


def _soften_overstrict_action(proposed: HDIAction, parsed: dict[str, Any]) -> HDIAction:
    """Avoid treating weak/unsupported evidence as hard contradiction.

    The verifier should be a calibrated reviewer, not a gate that refutes a
    hypothesis just because a finding is imperfectly supported. REFUTE is kept
    only when at least one checked claim is explicitly CONTRADICTED.
    """
    if proposed != HDIAction.REFUTE:
        return proposed
    claims = parsed.get("claim_verifications") or []
    has_contradiction = any(
        str(c.get("status", "")).upper() == "CONTRADICTED"
        for c in claims
        if isinstance(c, dict)
    )
    return HDIAction.REFUTE if has_contradiction else HDIAction.STOP


def _status_for_finding(index: int, evidence_chain: str, verifier_claims: list[dict[str, Any]]) -> str:
    """Map verifier claim statuses back to a parsed Investigator finding."""
    if not verifier_claims:
        return "SUPPORTED"

    for claim in verifier_claims:
        if not isinstance(claim, dict):
            continue
        if claim.get("finding_index") == index:
            return _clean_status(claim.get("status"))

    evidence_norm = evidence_chain.strip().lower()
    if evidence_norm:
        for claim in verifier_claims:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim", "")).strip().lower()
            if claim_text and (claim_text in evidence_norm or evidence_norm in claim_text):
                return _clean_status(claim.get("status"))

    return "SUPPORTED"


def _clean_status(raw: Any) -> str:
    status = str(raw or "SUPPORTED").upper()
    if status in {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}:
        return status
    return "SUPPORTED"


def _claim_status_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}
    for claim in claims:
        if not isinstance(claim, dict):
            counts["UNKNOWN"] += 1
            continue
        status = str(claim.get("status", "")).upper()
        if status in counts and status != "UNKNOWN":
            counts[status] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts


def _default_prompt_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / filename
