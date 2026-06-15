"""CPEG — Cross-Post Evidence Graph (revision §5).

A typed graph that links evidence to hypotheses across posts, enabling
inference chaining (e.g., job=ByteDance + city=Beijing → workplace_area=后厂村).

Why this is a contribution:
  - Privacy investigation needs cross-post linkage; geolocation / generic
    agent benchmarks treat each task independently.
  - Inference chaining is privacy-specific — combining established hypotheses
    yields new ones not visible from any single post.

Design choices:
  - Pure Python data structure, no external graph library (keeps it simple,
    serializable, and dependency-free).
  - All operations are O(N) over relevant subsets — fine for ≤500 posts.
  - Edges carry explicit weights so conflict resolution can be deterministic.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from argus.types import Evidence, Hypothesis


# ── Edge types ──

@dataclass
class CiteEdge:
    """Evidence is sourced from a particular post."""
    evidence_id: str
    post_id: str


@dataclass
class SupportEdge:
    """Evidence supports a hypothesis with a weight in [0, 1]."""
    evidence_id: str
    hypothesis_id: str
    weight: float
    note: str = ""


@dataclass
class RefuteEdge:
    """Evidence refutes a hypothesis with a weight in [0, 1]."""
    evidence_id: str
    hypothesis_id: str
    weight: float
    note: str = ""


@dataclass
class ChainEdge:
    """Hypothesis A combined with B yields/constrains hypothesis C."""
    src_a: str
    src_b: str
    dst: str
    rule: str = ""              # short string explaining the chaining rule


@dataclass
class PostNode:
    post_id: str
    timestamp: str | None = None
    perceptual_signature: dict[str, Any] = field(default_factory=dict)


# ── Main graph ──

class CPEG:
    """Cross-Post Evidence Graph.

    Stores PostNodes, EvidenceNodes, HypothesisNodes, and four edge types.
    Provides query methods for HDI / Profile synthesis.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.posts: dict[str, PostNode] = {}
        self.evidence: dict[str, Evidence] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.cites: list[CiteEdge] = []
        self.supports: list[SupportEdge] = []
        self.refutes: list[RefuteEdge] = []
        self.chains: list[ChainEdge] = []

    # ── Node insertion ──

    def add_post(
        self,
        post_id: str,
        *,
        timestamp: str | None = None,
        perceptual_signature: dict[str, Any] | None = None,
    ) -> PostNode:
        node = PostNode(
            post_id=post_id,
            timestamp=timestamp,
            perceptual_signature=perceptual_signature or {},
        )
        self.posts[post_id] = node
        return node

    def add_evidence(self, evidence: Evidence) -> str:
        """Add an evidence node and auto-create a cite edge."""
        if not evidence.id:
            evidence.id = str(uuid.uuid4())
        self.evidence[evidence.id] = evidence
        self.cites.append(CiteEdge(evidence_id=evidence.id, post_id=evidence.post_id))
        return evidence.id

    def add_hypothesis(self, h: Hypothesis) -> str:
        if not h.id:
            h.id = str(uuid.uuid4())
        self.hypotheses[h.id] = h
        return h.id

    # ── Edge insertion ──

    def link_support(
        self,
        evidence_id: str,
        hypothesis_id: str,
        *,
        weight: float = 1.0,
        note: str = "",
    ) -> None:
        self.supports.append(
            SupportEdge(evidence_id=evidence_id, hypothesis_id=hypothesis_id, weight=weight, note=note)
        )
        h = self.hypotheses.get(hypothesis_id)
        if h is not None and evidence_id not in h.supporting_evidence_ids:
            h.supporting_evidence_ids.append(evidence_id)

    def link_refute(
        self,
        evidence_id: str,
        hypothesis_id: str,
        *,
        weight: float = 1.0,
        note: str = "",
    ) -> None:
        self.refutes.append(
            RefuteEdge(evidence_id=evidence_id, hypothesis_id=hypothesis_id, weight=weight, note=note)
        )
        h = self.hypotheses.get(hypothesis_id)
        if h is not None and evidence_id not in h.refuting_evidence_ids:
            h.refuting_evidence_ids.append(evidence_id)

    def link_chain(self, src_a: str, src_b: str, dst: str, rule: str = "") -> None:
        self.chains.append(ChainEdge(src_a=src_a, src_b=src_b, dst=dst, rule=rule))
        for src in (src_a, src_b):
            ha, hb = self.hypotheses.get(src), self.hypotheses.get(dst)
            if ha is not None and hb is not None and dst not in ha.related_hypothesis_ids:
                ha.related_hypothesis_ids.append(dst)

    # ── Queries used by HDI / Routing-Verifier / Profile ──

    def active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.is_active]

    def hypotheses_for(self, attribute: str, *, only_active: bool = True) -> list[Hypothesis]:
        hs = [h for h in self.hypotheses.values() if h.attribute == attribute]
        return [h for h in hs if h.is_active] if only_active else hs

    def evidence_for_hypothesis(self, hypothesis_id: str) -> tuple[list[Evidence], list[Evidence]]:
        """Return (supporting_evidence, refuting_evidence) for a hypothesis."""
        s_ids = {e.evidence_id for e in self.supports if e.hypothesis_id == hypothesis_id}
        r_ids = {e.evidence_id for e in self.refutes if e.hypothesis_id == hypothesis_id}
        sup = [self.evidence[i] for i in s_ids if i in self.evidence]
        ref = [self.evidence[i] for i in r_ids if i in self.evidence]
        return sup, ref

    def support_score(self, hypothesis_id: str) -> float:
        """Sum of support weights minus refute weights, clipped to [0, ∞)."""
        s = sum(e.weight for e in self.supports if e.hypothesis_id == hypothesis_id)
        r = sum(e.weight for e in self.refutes if e.hypothesis_id == hypothesis_id)
        return max(0.0, s - r)

    def conflicting_pairs(self, attribute: str) -> list[tuple[str, str]]:
        """Return active hypothesis ID pairs (a, b) for the same attribute with
        clearly distinct values — candidates for BRANCH or merger.
        """
        active = self.hypotheses_for(attribute, only_active=True)
        out: list[tuple[str, str]] = []
        for i, ha in enumerate(active):
            for hb in active[i + 1:]:
                if ha.value.strip().lower() != hb.value.strip().lower():
                    out.append((ha.id, hb.id))
        return out

    # ── Inference chaining (revision §5.3) ──

    # Privacy-specific chaining rules. Each rule is a function:
    #   (cpeg) -> list[ChainEdge]   (new chains to register)
    # The rule should also append a derived hypothesis to the graph if needed.

    def apply_chaining_rules(self) -> list[ChainEdge]:
        """Apply built-in chaining rules and return newly created chain edges.

        Rules (extensible):
          R1: occupation.role + location.work-city → workplace_area hint
          R2: education.institution + location.city → student_area hint
          R3: lifestyle.travel + financial.consumption → income lower bound
        """
        new_edges: list[ChainEdge] = []
        new_edges.extend(self._chain_rule_workplace_area())
        new_edges.extend(self._chain_rule_student_area())
        return new_edges

    def _chain_rule_workplace_area(self) -> list[ChainEdge]:
        """If we have occupation.role (with org) + location.work at city level,
        record a chain pointing to a finer location.work hypothesis (district hint).
        """
        out: list[ChainEdge] = []
        roles = [h for h in self.hypotheses_for("occupation.role") if h.confidence >= 0.6]
        works = [h for h in self.hypotheses_for("location.work") if h.confidence >= 0.5]
        for role in roles:
            for work in works:
                # Only chain if role looks like role+org and work is at city level (level=2 of location)
                if "@" in role.value or "at " in role.value.lower() or "于" in role.value:
                    if work.level >= 2 and not self._chain_exists(role.id, work.id):
                        out.append(ChainEdge(
                            src_a=role.id, src_b=work.id, dst=work.id,
                            rule="role+city → narrow workplace district",
                        ))
        self.chains.extend(out)
        return out

    def _chain_rule_student_area(self) -> list[ChainEdge]:
        out: list[ChainEdge] = []
        edus = [h for h in self.hypotheses_for("education.institution") if h.confidence >= 0.6]
        homes = [h for h in self.hypotheses_for("location.home") if h.confidence >= 0.4]
        for edu in edus:
            for home in homes:
                if home.level <= 2 and not self._chain_exists(edu.id, home.id):
                    out.append(ChainEdge(
                        src_a=edu.id, src_b=home.id, dst=home.id,
                        rule="institution+city → student housing area",
                    ))
        self.chains.extend(out)
        return out

    def _chain_exists(self, a: str, b: str) -> bool:
        for c in self.chains:
            if {c.src_a, c.src_b} == {a, b}:
                return True
        return False

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "posts": {pid: asdict(p) for pid, p in self.posts.items()},
            "evidence": {eid: asdict(e) for eid, e in self.evidence.items()},
            "hypotheses": {hid: asdict(h) for hid, h in self.hypotheses.items()},
            "cites":    [asdict(e) for e in self.cites],
            "supports": [asdict(e) for e in self.supports],
            "refutes":  [asdict(e) for e in self.refutes],
            "chains":   [asdict(e) for e in self.chains],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)
