"""ARGUSPipeline — per-user orchestration.

Outer loop : for each post → ARGUS-Loop (per-post investigation).
Inner loop : alternating Investigator ↔ Routing-Verifier until HDI termination.

Final     : project CPEG → UserProfile (deterministic, no LLM).

Public API:

    pipeline = ARGUSPipeline.build_default(...)
    profile, cpeg = pipeline.run_user(user_id, user_metadata, posts)

`posts` is a list of dicts with at least:
    {"post_id": str, "caption": str, "media_files": [path, ...],
     "timestamp": str | None, "location_ip": str | None}

`user_metadata` is a free dict; only used for logging today.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from argus.types import (
    PerceptualSignature,
    Hypothesis,
    HDIAction,
    RouteDecision,
    UserProfile,
)
from argus.cpeg import CPEG
from argus.hypothesis import HypothesisPool
from argus.routing.amtr import AMTR
from argus.components.perception import Perception
from argus.components.investigator import Investigator, StepResult
from argus.components.routing_verifier import RoutingVerifier
from argus.profile import synthesize_full
from argus.metrics import start_post, finish_post, set_current_post
from argus.logging_utils import argus_log

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    argus_log(message)


# ── Config ──

@dataclass
class PipelineConfig:
    """Tunable budgets and limits."""
    max_posts_per_user: int = 50     # safety guard
    seed_discovery: bool = True      # if no hypothesis exists, do a discovery step


# ── Per-step log record ──

@dataclass
class StepLog:
    post_id: str
    step_id: int
    target_hypothesis_id: str | None
    route: RouteDecision | None
    investigator: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)


# ── Pipeline ──

class ARGUSPipeline:
    """Wires Perception, Investigator, Routing-Verifier, AMTR, HDI, CPEG."""

    def __init__(
        self,
        *,
        perception: Perception,
        investigator: Investigator,
        routing_verifier: RoutingVerifier,
        amtr: AMTR,
        config: PipelineConfig | None = None,
        narrative_runner=None,
    ):
        self.perception = perception
        self.investigator = investigator
        self.routing_verifier = routing_verifier
        self.amtr = amtr
        self.config = config or PipelineConfig()
        self.narrative_runner = narrative_runner       # Layer 2 (optional)

    # ── Convenience builder ──

    @classmethod
    def build_hybrid_default(
        cls,
        *,
        config: PipelineConfig | None = None,
        enable_narrative: bool = True,
        enable_delegation: bool = True,
    ) -> "ARGUSPipeline":
        """Build the ARGUS pipeline using the hybrid LLM backend stack.

          Investigator (backbone)         : GPT-5.5 via OpenAI-compatible router
          Profile narrative (Layer 2)     : GPT-5.5 via OpenAI-compatible router
          Routing-Verifier                : Qwen3.7-Max via DashScope by default
          Perception VL                   : Qwen3.6-Plus      via DashScope
          AMTR LLM fallback               : Qwen3.7-Max       via DashScope
          Visual delegation (deep_visual): Gemini 3.1 Pro    via Vertex AI native

        Requires:
          - ARGUS_ROUTER_API_KEY (backbone router)
          - GOOGLE_KEY_PATH     (Gemini Vertex AI service account)
          - DASHSCOPE_API_KEY   (Qwen)
          - SERPAPI_API_KEY / AMAP_API_KEY  (tools)
          - HF_TOKEN            (PaddleOCR model download)
        """
        # ── Backbone router side (OpenAI-compatible) ──
        from argus.openrouter import (
            make_investigator_runner_openrouter,
            make_narrative_runner_openrouter,
        )
        # ── Gemini + Qwen + OCR side (native APIs) ──
        from argus.adapters import (
            make_vl_fn_qwen_dashscope,
            make_amtr_fallback_qwen_dashscope,
            make_delegation_tools_gemini,
            make_verifier_runner,
            make_ocr_fn,
        )

        # Visual delegation tool (Gemini-native) is passed *into* the backbone
        # Investigator so it can call them via OpenAI tool-call protocol.
        delegation_tools = make_delegation_tools_gemini() if enable_delegation else None

        amtr = AMTR(llm_fallback=make_amtr_fallback_qwen_dashscope())
        perception = Perception(
            ocr_fn=make_ocr_fn(),
            vl_fn=make_vl_fn_qwen_dashscope(),
        )
        investigator = Investigator(
            agent_runner=make_investigator_runner_openrouter(
                delegation_tools=delegation_tools,
            ),
        )
        rv = RoutingVerifier(
            llm_runner=make_verifier_runner(),
            amtr=amtr,
        )
        narrative = make_narrative_runner_openrouter() if enable_narrative else None
        return cls(
            perception=perception,
            investigator=investigator,
            routing_verifier=rv,
            amtr=amtr,
            config=config,
            narrative_runner=narrative,
        )

    # ── Backwards-compat alias ──
    build_default = build_hybrid_default

    # ── Public entry point ──

    def run_user(
        self,
        user_id: str,
        user_metadata: dict[str, Any],
        posts: list[dict[str, Any]],
    ) -> tuple[UserProfile, CPEG, list[StepLog]]:
        cpeg = CPEG(user_id=user_id)
        pool = HypothesisPool()
        all_logs: list[StepLog] = []
        self._current_user_id = user_id
        _progress(
            f"user={user_id} start posts={len(posts[: self.config.max_posts_per_user])}"
        )

        # Cap post count for safety
        capped_posts = posts[: self.config.max_posts_per_user]
        for idx, post in enumerate(capped_posts, start=1):
            _progress(f"user={user_id} post {idx}/{len(capped_posts)} id={post.get('post_id') or post.get('id')}")
            logs = self._run_post(post, cpeg=cpeg, pool=pool)
            all_logs.extend(logs)

        _progress(f"user={user_id} synthesizing final profile")
        profile = synthesize_full(
            cpeg,
            metadata={
                "user_id": user_id,
                "user_metadata": user_metadata,
                "n_posts_processed": len(all_logs),
            },
            narrative_runner=self.narrative_runner,
        )
        _progress(f"user={user_id} complete steps={len(all_logs)}")
        return profile, cpeg, all_logs

    # ── Per-post loop ──

    def _run_post(
        self,
        post: dict[str, Any],
        *,
        cpeg: CPEG,
        pool: HypothesisPool,
    ) -> list[StepLog]:
        post_id = str(post.get("post_id") or post.get("id") or f"post-{len(cpeg.posts)}")
        media_files = post.get("media_files", []) or []
        start_post(post_id, image_count=len(media_files))
        post_started = time.time()
        _progress(f"post={post_id} start images={len(media_files)}")

        # 1. Perception
        _progress(f"post={post_id} perception start")
        signature = self._load_or_run_perception(
            post_id=post_id,
            caption=post.get("caption", ""),
            media_files=media_files,
            timestamp=post.get("timestamp"),
        )
        _progress(
            f"post={post_id} perception done tag={signature.vl_tag or 'none'} "
            f"images={len(signature.image_summaries) or signature.image_count} "
            f"entities={len(signature.entities)}"
        )
        cpeg.add_post(
            post_id=post_id,
            timestamp=post.get("timestamp"),
            perceptual_signature={
                "vl_tag": signature.vl_tag,
                "vl_caption": signature.vl_caption,
                "image_summaries": signature.image_summaries,
                "entities": signature.entities,
                "image_count": signature.image_count,
            },
        )

        # 2. Pick a starting target hypothesis (or discovery mode)
        target = self._pick_initial_target(pool, signature)

        # 3. Decide initial route
        route = self.amtr.decide(
            signature=signature,
            hypothesis=target,
            active_attributes=pool.attributes_with_active(),
        )
        _progress(
            f"post={post_id} initial route model={route.model} tool_family={route.tool_family}"
        )

        logs: list[StepLog] = []
        try:
            # 4. Loop until STOP / budget exhaustion
            step_id = 0
            while True:
                if step_id >= 100:
                    _progress(f"post={post_id} hard safety guard hit at 100 steps")
                    break

                attribute_for_charge = (
                    target.attribute if target is not None else "behavioral.personality"
                )

                step_id += 1
                _progress(
                    f"post={post_id} step={step_id} investigator start "
                    f"attr={attribute_for_charge} route={route.model}/{route.tool_family}"
                )

                # Investigator
                prior = _slice_evidence_for(cpeg, attribute_for_charge, limit=8)
                step_result: StepResult = self.investigator.run_step(
                    post=post,
                    signature=signature,
                    target_hypothesis=target,
                    route=route,
                    prior_evidence=prior,
                )
                _progress(
                    f"post={post_id} step={step_id} investigator done "
                    f"findings={len(step_result.findings)} "
                    f"tools={step_result.tool_call_count}"
                )

                # Routing-Verifier (verifies, updates pool, decides next)
                _progress(f"post={post_id} step={step_id} verifier start")
                v_result = self.routing_verifier.verify_and_route(
                    signature=signature,
                    target_hypothesis=target,
                    investigator_output=step_result.raw_output,
                    investigator_findings=step_result.findings,
                    prior_evidence=prior,
                    pool=pool,
                    cpeg=cpeg,
                    post_id=post_id,
                    step_id=step_id,
                )
                _progress(
                    f"post={post_id} step={step_id} verifier done "
                    f"action={v_result.action.value} new_evidence={len(v_result.new_evidence)} "
                    f"updates={len(v_result.hypothesis_updates)}"
                )

                logs.append(StepLog(
                    post_id=post_id,
                    step_id=step_id,
                    target_hypothesis_id=target.id if target else None,
                    route=route,
                    investigator={
                        "n_findings": len(step_result.findings),
                        "n_tools": step_result.tool_call_count,
                        "suggested_action": step_result.suggested_action,
                        "rationale": step_result.rationale,
                        "raw_output": step_result.raw_output,
                        "findings": step_result.findings,
                    },
                    verifier={
                        "action": v_result.action.value,
                        "uncertainty": v_result.uncertainty_score,
                        "n_hyp_updates": len(v_result.hypothesis_updates),
                        "n_new_evidence": len(v_result.new_evidence),
                        "rationale": v_result.rationale,
                        "raw_output": v_result.raw_output,
                        "new_evidence": [asdict(e) for e in v_result.new_evidence],
                    },
                ))

                # Loop control. REFUTE means this route/hypothesis is closed for
                # the current post; without this break, the verifier can keep
                # refuting the same target indefinitely.
                if v_result.action in (HDIAction.STOP, HDIAction.REFUTE):
                    _progress(
                        f"post={post_id} {v_result.action.value} requested by verifier; "
                        "stopping post"
                    )
                    break

                # Update target / route for next iteration
                target = (
                    pool.get(v_result.target_hypothesis_id)
                    if v_result.target_hypothesis_id
                    else target
                )
                route = v_result.next_route or self.amtr.decide(
                    signature=signature,
                    hypothesis=target,
                    active_attributes=pool.attributes_with_active(),
                )
        finally:
            finish_post(
                post_id,
                step_count=len(logs),
                tool_call_count=sum(int(l.investigator.get("n_tools", 0)) for l in logs),
            )
            set_current_post(None)
            _progress(f"post={post_id} complete steps={len(logs)} elapsed={time.time() - post_started:.1f}s")

        return logs

    def _load_or_run_perception(
        self,
        *,
        post_id: str,
        caption: str,
        media_files: list[str],
        timestamp: str | None,
    ) -> PerceptualSignature:
        user_id = getattr(self, "_current_user_id", "unknown_user")
        cache_root = os.environ.get("ARGUS_CACHE_DIR", "outputs/argus/cache")
        cache_dir = Path(cache_root) / "perception" / user_id
        cache_path = cache_dir / f"{post_id}.json"
        key = _perception_cache_key(caption, media_files, timestamp)
        if os.environ.get("ARGUS_DISABLE_CACHE", "0") != "1" and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("cache_key") == key:
                    sig = PerceptualSignature(**payload["signature"])
                    _progress(f"post={post_id} perception cache hit {cache_path}")
                    return sig
                _progress(f"post={post_id} perception cache stale; recomputing")
            except Exception as exc:
                logger.warning("Failed reading perception cache %s: %s", cache_path, exc)

        sig = self.perception.process(
            post_id=post_id,
            caption=caption,
            media_files=media_files,
            timestamp=timestamp,
        )
        if os.environ.get("ARGUS_DISABLE_CACHE", "0") != "1":
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "user_id": user_id,
                        "post_id": post_id,
                        "signature": asdict(sig),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _progress(f"post={post_id} perception cached {cache_path}")
        return sig

    # ── Initial target selection ──

    def _pick_initial_target(
        self,
        pool: HypothesisPool,
        signature: PerceptualSignature,
    ) -> Hypothesis | None:
        """Pick the most promising active hypothesis for this post.

        Heuristic: prefer non-converged active hypotheses whose attribute_class
        matches the perceptual VL tag (e.g. tag=navigation → location.* family).
        Falls back to highest sensitivity unconverged attribute, else None
        (which puts the loop into discovery mode).
        """
        active = [h for h in pool.active() if not pool.is_converged(h.id)]
        if not active:
            return None

        # Priority 1: tag-aligned hypothesis
        tag_to_class = {
            "navigation": "location",
            "landmark": "location",
            "signage": "location",
            "id_card": "identity",
            "document": "identity",
            "wedding": "social",
            "graduation": "education",
            "school": "education",
            "vehicle": "financial",
            "luxury": "financial",
            "product": "financial",
        }
        target_class = tag_to_class.get(signature.vl_tag)
        if target_class:
            tagged = [h for h in active if h.attribute.startswith(target_class + ".")]
            if tagged:
                tagged.sort(key=lambda h: h.confidence, reverse=True)
                return tagged[0]

        # Priority 2: highest-sensitivity attribute that has unconverged hypotheses
        from argus.attributes import get_sensitivity
        active.sort(key=lambda h: (get_sensitivity(h.attribute), h.confidence), reverse=True)
        return active[0]


# ── Convenience wrapper ──

def run_user(
    user_id: str,
    user_metadata: dict[str, Any],
    posts: list[dict[str, Any]],
    *,
    config: PipelineConfig | None = None,
) -> tuple[UserProfile, CPEG, list[StepLog]]:
    """One-call entry: build default pipeline and run."""
    pipeline = ARGUSPipeline.build_default(config=config)
    return pipeline.run_user(user_id, user_metadata, posts)


# ── Helpers ──

def _slice_evidence_for(cpeg: CPEG, attribute: str, *, limit: int = 8):
    """Return evidence linked to any hypothesis of this attribute, latest first."""
    hyps = cpeg.hypotheses_for(attribute, only_active=False)
    out = []
    seen = set()
    for h in hyps:
        sup, ref = cpeg.evidence_for_hypothesis(h.id)
        for e in sup + ref:
            if e.id in seen:
                continue
            seen.add(e.id)
            out.append(e)
    return out[-limit:][::-1]


def _perception_cache_key(caption: str, media_files: list[str], timestamp: str | None) -> str:
    payload = json.dumps(
        {
            "caption": caption,
            "media_files": media_files,
            "timestamp": timestamp,
            "qwen_image_max_side": os.environ.get("ARGUS_QWEN_PERCEPTION_MAX_SIDE", "768"),
            "perception_prompt_version": "per-image-v1",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
