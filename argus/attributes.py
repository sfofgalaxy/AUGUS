"""Attribute taxonomy + sensitivity for ARGUS / EGHE.

Defines the 28 attribute slots, their granularity hierarchy (taxonomy tree),
sensitivity score (1-5), and difficulty seed. Used by:

  - HDI       : decide which hypotheses to track
  - CPEG      : default taxonomy when adding new hypothesis
  - EGHE-Tier1: granularity scoring at evaluation time

This file is the single source of truth for attribute schema. If you add a
new attribute, also update the EGHE evaluator.
"""
from __future__ import annotations


# ── Granularity ladders (revision §1.3, EGHE Tier 1) ──
# Each ladder is ordered coarsest → finest. Score per level is interpolated.

_LADDERS: dict[str, list[str]] = {
    "location":      ["country", "province", "city", "district", "exact"],
    "occupation":    ["sector", "role_type", "specific_role", "role_org", "full"],
    "education":     ["has_degree", "degree_level", "institution", "department", "full"],
    "age":           ["generation", "decade", "5yr_range", "exact"],
    "income":        ["qualitative", "bracket", "specific"],
    "name":          ["partial", "full"],
    "binary":        ["unknown", "value"],                    # gender, marital, etc.
    "free":          ["unknown", "category", "specific"],     # hobbies, brands, etc.
}


def _scores(n: int) -> list[float]:
    """Linear interpolation from 0.3 to 1.0 across n levels (matches EGHE Tier 1)."""
    if n <= 1:
        return [1.0]
    return [round(0.3 + 0.7 * i / (n - 1), 2) for i in range(n)]


def _ladder(name: str) -> tuple[list[str], list[float]]:
    levels = _LADDERS[name]
    return levels, _scores(len(levels))


# ── 28 attribute registry ──
# Sensitivity scale (1-5): 1=trivial, 5=maximally sensitive (PII / finance / location-precise)
# Difficulty seed (1-4): 1=usually surface evidence, 4=requires multi-post chaining

_REGISTRY: dict[str, dict] = {
    # ── Identity (6) ──
    "identity.name":            {"ladder": "name",       "sensitivity": 5, "difficulty": 3},
    "identity.age":             {"ladder": "age",        "sensitivity": 3, "difficulty": 2},
    "identity.gender":          {"ladder": "binary",     "sensitivity": 2, "difficulty": 1},
    "identity.nationality":     {"ladder": "free",       "sensitivity": 3, "difficulty": 2},
    "identity.ethnicity":       {"ladder": "free",       "sensitivity": 4, "difficulty": 3},
    "identity.id_number":       {"ladder": "binary",     "sensitivity": 5, "difficulty": 4},

    # ── Location (4) ──
    "location.home":            {"ladder": "location",   "sensitivity": 5, "difficulty": 4},
    "location.work":            {"ladder": "location",   "sensitivity": 4, "difficulty": 3},
    "location.hometown":        {"ladder": "location",   "sensitivity": 3, "difficulty": 2},
    "location.frequent_areas":  {"ladder": "location",   "sensitivity": 4, "difficulty": 3},

    # ── Social (5) ──
    "social.relationship":      {"ladder": "free",       "sensitivity": 3, "difficulty": 2},
    "social.partner":           {"ladder": "name",       "sensitivity": 4, "difficulty": 3},
    "social.children":          {"ladder": "free",       "sensitivity": 4, "difficulty": 3},
    "social.family":            {"ladder": "free",       "sensitivity": 3, "difficulty": 3},
    "social.friends":           {"ladder": "free",       "sensitivity": 2, "difficulty": 2},

    # ── Financial / Education / Work (5) ──
    "financial.income":         {"ladder": "income",     "sensitivity": 4, "difficulty": 3},
    "financial.consumption":    {"ladder": "free",       "sensitivity": 2, "difficulty": 2},
    "financial.assets":         {"ladder": "free",       "sensitivity": 5, "difficulty": 4},
    "education.institution":    {"ladder": "education",  "sensitivity": 3, "difficulty": 2},
    "occupation.role":          {"ladder": "occupation", "sensitivity": 3, "difficulty": 2},

    # ── Health / Lifestyle (4) ──
    "health.condition":         {"ladder": "free",       "sensitivity": 5, "difficulty": 3},
    "health.fitness":           {"ladder": "free",       "sensitivity": 2, "difficulty": 1},
    "lifestyle.hobbies":        {"ladder": "free",       "sensitivity": 1, "difficulty": 1},
    "lifestyle.travel":         {"ladder": "location",   "sensitivity": 2, "difficulty": 2},

    # ── Behavioral / Inferred (4) ──
    "behavioral.personality":   {"ladder": "free",       "sensitivity": 2, "difficulty": 3},
    "behavioral.political":     {"ladder": "free",       "sensitivity": 5, "difficulty": 3},
    "behavioral.religion":      {"ladder": "free",       "sensitivity": 4, "difficulty": 3},
    "behavioral.sexuality":     {"ladder": "free",       "sensitivity": 5, "difficulty": 4},
}


# ── Public API ──

ALL_ATTRIBUTES: tuple[str, ...] = tuple(_REGISTRY.keys())


def get_levels(attribute: str) -> list[str]:
    """Return the ordered taxonomy levels for an attribute."""
    ladder = _REGISTRY.get(attribute, {"ladder": "free"})["ladder"]
    return _LADDERS[ladder]


def get_max_level(attribute: str) -> int:
    """Index of the deepest (most precise) level (0-based)."""
    return len(get_levels(attribute)) - 1


def level_score(attribute: str, level_idx: int) -> float:
    """EGHE Tier 1 score for inferring an attribute up to level_idx."""
    if level_idx < 0:
        return 0.0
    ladder = _REGISTRY.get(attribute, {"ladder": "free"})["ladder"]
    _, scores = _ladder(ladder)
    return scores[min(level_idx, len(scores) - 1)]


def get_sensitivity(attribute: str) -> int:
    """Sensitivity score in [1, 5]. Used for ranking/evaluation metadata."""
    return int(_REGISTRY.get(attribute, {"sensitivity": 1})["sensitivity"])


def get_difficulty(attribute: str) -> int:
    """Default difficulty seed in [1, 4] (overridable per-instance)."""
    return int(_REGISTRY.get(attribute, {"difficulty": 1})["difficulty"])


def attribute_class(attribute: str) -> str:
    """Top-level class (identity / location / social / ...)."""
    return attribute.split(".", 1)[0]


def all_classes() -> list[str]:
    return sorted({attribute_class(a) for a in ALL_ATTRIBUTES})
