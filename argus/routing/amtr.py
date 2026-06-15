"""AMTR — Adaptive Model-Tool Routing (revision §6).

Per-step decision: given (perceptual_signature, target_hypothesis, region_hint)
return (model, tool_family). Two paths:

  Fast path : YAML routing-table lookup (deterministic, cheap, auditable).
  Slow path : LLM fallback when no rule matches.

The fast path is the primary contribution. The table is task-aware because
its conditioning includes both the *perceptual signature* (what was found in
this post) and the *target hypothesis attribute* (what HDI wants to advance).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from argus.attributes import attribute_class
from argus.types import (
    PerceptualSignature,
    Hypothesis,
    RouteDecision,
    TOOL_FAMILIES,
)

logger = logging.getLogger(__name__)

_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "routing_table.yaml"


# ── Routing table abstraction ──

@dataclass
class _Rule:
    name: str
    when: dict[str, Any]
    route: dict[str, Any]


class RoutingTable:
    """In-memory representation of routing_table.yaml."""

    def __init__(self, rules: list[_Rule], default: dict[str, Any]):
        self.rules = rules
        self.default = default

    @classmethod
    def load(cls, path: Path | None = None) -> "RoutingTable":
        path = path or _DEFAULT_TABLE_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rules = [
            _Rule(name=r.get("name", f"rule_{i}"), when=r.get("when", {}), route=r.get("route", {}))
            for i, r in enumerate(data.get("rules", []))
        ]
        default = data.get("default") or {
            "model": "auto", "tool_family": "web_search", "rationale": "default",
        }
        return cls(rules=rules, default=default)


# ── AMTR ──

class AMTR:
    """Adaptive Model-Tool Routing — fast path + optional LLM fallback.

    Args:
      table : a RoutingTable instance (loaded once, reused per call).
      llm_fallback : optional callable (state_dict) -> dict({"model", "tool_family", "rationale"}).
                     If None, uses the table's `default` block on miss.
      region_resolver : optional callable (caption_text) -> "cn" / "overseas" / "unknown".
                        Defaults to a simple keyword scanner.
    """

    def __init__(
        self,
        *,
        table: RoutingTable | None = None,
        llm_fallback=None,
        region_resolver=None,
    ):
        self.table = table or RoutingTable.load()
        self.llm_fallback = llm_fallback
        self.region_resolver = region_resolver or _default_region_resolver

    # ── Public decision API ──

    def decide(
        self,
        signature: PerceptualSignature,
        hypothesis: Hypothesis | None,
        *,
        region_hint: str | None = None,
        active_attributes: list[str] | None = None,
    ) -> RouteDecision:
        """Return a RouteDecision for the next investigation step."""
        attr = hypothesis.attribute if hypothesis is not None else None
        cls = attribute_class(attr) if attr else None

        region = region_hint or self.region_resolver(signature.raw_post_text or signature.vl_caption or "")

        # ── Fast path: table lookup ──
        for rule in self.table.rules:
            if self._matches(rule.when, attr=attr, cls=cls, sig=signature, region=region):
                model = rule.route.get("model", "auto")
                tool = rule.route.get("tool_family", "web_search")
                rationale = rule.route.get("rationale", rule.name)
                if tool not in TOOL_FAMILIES:
                    logger.warning("AMTR rule %r emitted unknown tool_family %r; skipping.", rule.name, tool)
                    continue
                return RouteDecision(
                    model=model,
                    tool_family=tool,
                    target_hypothesis_id=hypothesis.id if hypothesis else None,
                    rationale=f"[fast:{rule.name}] {rationale}",
                    fast_path=True,
                )

        # ── Slow path: LLM fallback ──
        if self.llm_fallback is not None:
            try:
                proposal = self.llm_fallback({
                    "attribute": attr,
                    "attribute_class": cls,
                    "perceptual_signature": signature,
                    "region": region,
                    "active_attributes": active_attributes or [],
                })
                if isinstance(proposal, dict) and proposal.get("tool_family") in TOOL_FAMILIES:
                    return RouteDecision(
                        model=proposal.get("model", "auto"),
                        tool_family=proposal["tool_family"],
                        target_hypothesis_id=hypothesis.id if hypothesis else None,
                        rationale=f"[llm] {proposal.get('rationale', '')}",
                        fast_path=False,
                    )
            except Exception as exc:
                logger.warning("AMTR LLM fallback failed: %s", exc)

        # ── Final default ──
        d = self.table.default
        return RouteDecision(
            model=d.get("model", "auto"),
            tool_family=d.get("tool_family", "web_search"),
            target_hypothesis_id=hypothesis.id if hypothesis else None,
            rationale=f"[default] {d.get('rationale', '')}",
            fast_path=False,
        )

    # ── Rule matching ──

    @staticmethod
    def _matches(
        when: dict[str, Any],
        *,
        attr: str | None,
        cls: str | None,
        sig: PerceptualSignature,
        region: str,
    ) -> bool:
        # `attribute` exact match
        w_attr = when.get("attribute")
        if w_attr is not None and w_attr != attr:
            return False
        # `attribute_class` match
        w_cls = when.get("attribute_class")
        if w_cls is not None and w_cls != cls:
            return False
        # `vl_tag` (string or list) — at least one match
        w_tag = when.get("vl_tag")
        if w_tag is not None:
            tags = [w_tag] if isinstance(w_tag, str) else list(w_tag)
            sig_tags = {sig.vl_tag or ""}
            sig_tags.update(
                str(item.get("vl_tag") or "")
                for item in getattr(sig, "image_summaries", [])
            )
            if not sig_tags.intersection(tags):
                return False
        # `entity_type` — entity dict must contain a non-empty value for that type
        w_ent = when.get("entity_type")
        if w_ent is not None:
            if not (sig.entities.get(w_ent) or []):
                return False
        # `region_hint`
        w_region = when.get("region_hint")
        if w_region is not None and w_region != region:
            return False
        return True


# ── Lightweight region resolver (kept here so AMTR is self-contained) ──

_CN_MARKERS = (
    "中国", "大陆", "北京", "上海", "广州", "深圳", "杭州", "南京", "成都",
    "重庆", "武汉", "西安", "天津", "苏州", "宁波", "青岛", "厦门", "长沙",
    "郑州", "合肥", "济南", "福州", "南宁", "昆明", "哈尔滨", "长春", "沈阳",
    "香港", "澳门", "台湾",
    "chengdu", "shanghai", "beijing", "shenzhen", "guangzhou", "hangzhou",
    "nanjing", "wuhan", "xian", "tianjin",
)
_OVERSEAS_MARKERS = (
    "usa", "united states", "uk", "london", "paris", "france", "germany",
    "japan", "tokyo", "korea", "seoul", "singapore",
    "new york", "los angeles", "san francisco",
)


def _default_region_resolver(text: str) -> str:
    if not text:
        return "unknown"
    low = text.lower()
    for kw in _CN_MARKERS:
        if kw in low:
            return "cn"
    for kw in _OVERSEAS_MARKERS:
        if kw in low:
            return "overseas"
    return "unknown"
