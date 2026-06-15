"""HDI — Hypothesis-Driven Investigation (revision §4).

HypothesisPool wraps the live working set of hypotheses for a single user.
It exposes the action set (CONTINUE / ESCALATE / REFUTE / BRANCH / STOP) as
explicit methods, and maintains confidence trajectories for ablation.

This module is a state holder, not an LLM caller — the Routing-Verifier (which
*is* an LLM caller, see components/routing_verifier.py) drives them.
"""
from __future__ import annotations

import uuid
from typing import Any

from argus.types import Hypothesis, HDIAction
from argus.attributes import get_max_level, get_levels


# ── HypothesisPool ──

class HypothesisPool:
    """Live set of hypotheses for one user during investigation.

    Note that hypotheses also live in CPEG; this pool is the *operational*
    handle for HDI actions, while CPEG is the persistent graph. They share
    the same Hypothesis instances by reference.
    """

    # Default thresholds (tunable in config)
    THRESH_HIGH_CONFIDENCE = 0.75
    THRESH_LOW_CONFIDENCE  = 0.20
    MAX_ACTIVE_PER_ATTR    = 3

    def __init__(self):
        self._by_id: dict[str, Hypothesis] = {}

    # ── Mutators (the HDI action verbs) ──

    def add(
        self,
        attribute: str,
        value: str,
        *,
        level: int = 0,
        confidence: float = 0.5,
        evidence_id: str | None = None,
    ) -> Hypothesis:
        """Create a new hypothesis. Caller is responsible for prerequisites
        (e.g. checking MAX_ACTIVE_PER_ATTR, registering with CPEG).
        """
        h = Hypothesis(
            id=str(uuid.uuid4()),
            attribute=attribute,
            value=value,
            level=level,
            max_level=get_max_level(attribute),
            confidence=confidence,
            supporting_evidence_ids=[evidence_id] if evidence_id else [],
        )
        h.history.append({"step": 0, "confidence": confidence, "reason": "initial"})
        self._by_id[h.id] = h
        return h

    def update_confidence(
        self,
        hypothesis_id: str,
        delta: float,
        *,
        step: int = 0,
        reason: str = "",
    ) -> None:
        h = self._by_id[hypothesis_id]
        h.confidence = max(0.0, min(1.0, h.confidence + delta))
        h.history.append({"step": step, "confidence": h.confidence, "reason": reason})

    def escalate(
        self,
        hypothesis_id: str,
        *,
        new_value: str,
        new_confidence: float | None = None,
        step: int = 0,
    ) -> Hypothesis | None:
        """ESCALATE: promote to next-finer level. Returns the *new* hypothesis
        (the original is frozen). Returns None if already terminal.
        """
        h = self._by_id[hypothesis_id]
        if h.is_terminal:
            return None
        h.status = "frozen"
        new_h = self.add(
            attribute=h.attribute,
            value=new_value,
            level=h.level + 1,
            confidence=new_confidence if new_confidence is not None else h.confidence,
        )
        new_h.related_hypothesis_ids.append(h.id)
        h.related_hypothesis_ids.append(new_h.id)
        new_h.history.append({"step": step, "confidence": new_h.confidence, "reason": f"escalated from {h.id}"})
        return new_h

    def refute(self, hypothesis_id: str, *, step: int = 0, reason: str = "") -> None:
        """REFUTE: mark hypothesis refuted; keeps it in graph for trace."""
        h = self._by_id[hypothesis_id]
        h.status = "refuted"
        h.history.append({"step": step, "confidence": h.confidence, "reason": f"refuted: {reason}"})

    def branch(
        self,
        hypothesis_id: str,
        *,
        new_value: str,
        new_confidence: float = 0.5,
        step: int = 0,
    ) -> Hypothesis:
        """BRANCH: introduce a competing hypothesis at the same level.
        Original hypothesis stays active; both compete via support score.
        """
        h = self._by_id[hypothesis_id]
        sib = self.add(
            attribute=h.attribute,
            value=new_value,
            level=h.level,
            confidence=new_confidence,
        )
        sib.related_hypothesis_ids.append(h.id)
        h.related_hypothesis_ids.append(sib.id)
        sib.history.append({"step": step, "confidence": new_confidence, "reason": f"branched from {h.id}"})
        return sib

    # ── Accessors ──

    def get(self, hypothesis_id: str) -> Hypothesis:
        return self._by_id[hypothesis_id]

    def all(self) -> list[Hypothesis]:
        return list(self._by_id.values())

    def active(self) -> list[Hypothesis]:
        return [h for h in self._by_id.values() if h.is_active]

    def active_for(self, attribute: str) -> list[Hypothesis]:
        return [h for h in self._by_id.values() if h.is_active and h.attribute == attribute]

    def attributes_with_active(self) -> list[str]:
        return sorted({h.attribute for h in self.active()})

    # ── HDI termination predicates (revision §4.2) ──

    def is_converged(self, hypothesis_id: str) -> bool:
        """A hypothesis is converged if it is at terminal level and high-confidence,
        or if it is unambiguously refuted (low confidence)."""
        h = self.get(hypothesis_id)
        if h.confidence >= self.THRESH_HIGH_CONFIDENCE and h.is_terminal:
            return True
        if h.confidence <= self.THRESH_LOW_CONFIDENCE:
            return True
        return False

    def all_active_converged(self) -> bool:
        """When True, the Routing-Verifier should STOP (no more useful work)."""
        return all(self.is_converged(h.id) for h in self.active())


# ── HDI helper: decide action given current state ──
#
# This is the *deterministic* portion of HDI. The LLM (in routing_verifier.py)
# proposes a desired action + hypothesis updates; this function validates and
# clamps the proposal against pool state.

def reconcile_action(
    proposed: HDIAction,
    *,
    target_hypothesis: Hypothesis | None,
    pool: HypothesisPool,
) -> HDIAction:
    """Validate the LLM-proposed HDI action against pool reality.

    Rules:
      - If all active converged → STOP.
      - If target_hypothesis is terminal and proposed is ESCALATE → coerce to STOP.
      - If too many active per attr and proposed is BRANCH → coerce to CONTINUE.
      - Otherwise → return proposed.
    """
    if pool.all_active_converged():
        return HDIAction.STOP
    if target_hypothesis is not None:
        if proposed == HDIAction.ESCALATE and target_hypothesis.is_terminal:
            return HDIAction.STOP
        if proposed == HDIAction.BRANCH:
            n_active = len(pool.active_for(target_hypothesis.attribute))
            if n_active >= HypothesisPool.MAX_ACTIVE_PER_ATTR:
                return HDIAction.CONTINUE
    return proposed
