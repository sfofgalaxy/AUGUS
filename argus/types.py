"""Core data types for ARGUS.

All structures are plain dataclasses — no LLM dependencies, no I/O. They flow
between Perception → Investigator → Routing-Verifier → CPEG → Profile.

Naming corresponds to revision.md formal definitions:
  - PerceptualSignature : §3 (Perception output, fed to AMTR)
  - Evidence            : §5 (CPEG node)
  - Hypothesis          : §4.2 (HDI hypothesis pool member)
  - RouteDecision       : §6.2 (AMTR output)
  - HDIAction           : §4.2 (state-machine action set)
  - UserProfile         : final hierarchical output
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── HDI state machine actions (revision §4.2) ──

class HDIAction(str, Enum):
    """Actions the Routing-Verifier may emit at each loop iteration."""
    CONTINUE = "continue"      # keep investigating current hypothesis
    ESCALATE = "escalate"      # promote hypothesis to next granularity level
    REFUTE = "refute"          # mark hypothesis refuted, drop from active set
    BRANCH = "branch"          # split into competing hypotheses (conflict)
    STOP = "stop"              # end this post's investigation


# ── Perception output (§3, fed to AMTR via signature) ──

@dataclass
class PerceptualSignature:
    """Compact summary of what perception found in a single post.

    Designed to be small (fit easily into prompts) but information-dense.
    """
    post_id: str = ""
    ocr_text: str = ""                                         # concat of OCR per image
    vl_caption: str = ""                                       # 1-line caption
    vl_tag: str = ""                                           # e.g. "navigation" / "wedding"
    image_summaries: list[dict[str, Any]] = field(default_factory=list)
    # one entry per image:
    #   {"image_index": 1, "path": "...", "vl_tag": "document",
    #    "vl_caption": "...", "entities": [...], "ocr_text": "..."}
    entities: dict[str, list[str]] = field(default_factory=dict)
    # entity types (string keys) → values:
    #   "address_fragments", "person_names", "brand_names",
    #   "model_numbers", "edu_keywords", "id_keywords",
    #   "event_keywords", "navigation_keywords"
    image_count: int = 0
    has_text_in_image: bool = False
    raw_post_text: str = ""                                    # original caption


# ── Evidence (CPEG node payload) ──

@dataclass
class Evidence:
    """A single piece of evidence collected during investigation.

    May come from perception (OCR/VL) or from a tool call (map/search/...).
    """
    id: str                                                    # uuid string
    text: str                                                  # serialized claim
    source_tool: str                                           # "ocr" | "vl_caption" | "amap_poi_search" | ...
    modality: str                                              # "text" | "image" | "ocr" | "tool_result"
    confidence: float                                          # [0, 1]
    post_id: str
    timestamp: str | None = None
    raw_data: dict[str, Any] | None = None                     # optional full tool output / verifier metadata


# ── Hypothesis (HDI pool member, §4.2) ──

@dataclass
class Hypothesis:
    """One hypothesis about a single attribute slot."""
    id: str                                                    # uuid
    attribute: str                                             # e.g. "location.home"
    value: str                                                 # e.g. "Beijing Haidian"
    level: int                                                 # current granularity level (0-indexed)
    max_level: int                                             # taxonomy max for this attribute
    confidence: float                                          # [0, 1]
    supporting_evidence_ids: list[str] = field(default_factory=list)
    refuting_evidence_ids: list[str] = field(default_factory=list)
    related_hypothesis_ids: list[str] = field(default_factory=list)
    status: str = "active"                                     # "active" | "refuted" | "frozen"
    history: list[dict[str, Any]] = field(default_factory=list)
    # confidence trajectory: each entry {"step": int, "confidence": float, "reason": str}

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_terminal(self) -> bool:
        """Whether the hypothesis has reached the deepest taxonomy level."""
        return self.level >= self.max_level


# ── AMTR routing decision (§6.2) ──

@dataclass
class RouteDecision:
    """One routing decision emitted by AMTR or Routing-Verifier."""
    model: str                                                 # "gemini" | "qwen" | "claude" | "gpt" | "auto"
    tool_family: str                                           # see TOOL_FAMILIES
    target_hypothesis_id: str | None = None
    rationale: str = ""
    fast_path: bool = False                                    # True if matched routing-table; False if LLM fallback


# ── Tool family canonical names (referenced from routing_table.yaml) ──

TOOL_FAMILIES: tuple[str, ...] = (
    "map_search",       # amap_poi_search / google_maps_search
    "web_search",       # google_search / bing_search
    "ocr",              # run_ocr (re-run on a crop)
    "zoom",             # crop_image / adaptive_zoom
    "fetch",            # fetch_webpage
    "stop",             # signal: do not investigate further this post
)


# ── Final user-level output ──

@dataclass
class UserProfile:
    """Hierarchical privacy profile, projected from CPEG after all posts.

    Two-layer design:
      Layer 1 (`attributes`)  : deterministic per-slot projection from CPEG.
                                Used by EGHE evaluation; reproducible.
      Layer 2 (`narrative`)   : LLM-synthesized cross-attribute insights,
                                implicit attributes, multi-value merges, and
                                a natural-language summary. Used for case
                                studies and human inspection.
    """
    user_id: str
    attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # keyed by attribute slot, e.g. "location.home" -> {
    #   "value": "...", "level_idx": 3, "level_name": "district",
    #   "confidence": 0.82, "evidence_ids": [...], "sensitivity": 5,
    #   "granularity_score": 0.85, ...
    # }
    narrative: dict[str, Any] = field(default_factory=dict)
    # {"cross_attribute_insights": [...], "implicit_attributes": [...],
    #  "merged_multi_value": [...], "narrative": "<paragraph>"}
    metadata: dict[str, Any] = field(default_factory=dict)
    # raw counts: total_posts, total_steps, mean_uncertainty, etc.
