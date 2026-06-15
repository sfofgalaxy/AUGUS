"""Perception — OCR + per-image VL captions.

Produces a PerceptualSignature consumed by Investigator and AMTR. Designed to
emit a compact data structure rather than a paragraph.

This module deliberately keeps the LLM/OCR plumbing thin: it receives
callables from `argus.adapters` and exposes a clean ARGUS-native interface.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from argus.types import PerceptualSignature

logger = logging.getLogger(__name__)


# ── Lightweight entity extractors (regex; cheap; deterministic) ──

_ADDRESS_RE = re.compile(
    r"[一-鿿]{1,8}(?:路|街|道|大道|广场|地铁站|高速|出口)"
    r"|[A-Z][a-z]+\s+(?:Road|Street|Avenue|Boulevard|Plaza|Square)"
)
_CN_NAME_RE = re.compile(r"[一-鿿]{2,3}")
_BRAND_RE = re.compile(r"\b(?:Apple|Samsung|Huawei|Xiaomi|Tesla|BMW|Audi|Mercedes|Louis Vuitton|Gucci|Chanel|Hermes)\b", re.I)
_MODEL_NUM_RE = re.compile(r"[A-Z]{2,}[\s-]?[A-Z0-9]{2,}|[A-Z][a-z]+\s(?:Pro|Max|Plus|Ultra|Mini)")
_EDU_KEYWORDS = ("大学", "学院", "university", "college", "PhD", "Master", "本科", "lab")
_ID_KEYWORDS = ("passport", "护照", "身份证", "id card", "certificate", "证书", "学位", "diploma", "签证", "visa")
_NAV_KEYWORDS = ("导航", "出口", "限速", "公里", "米后", "左转", "右转", "直行", "navi")
_EVENT_KEYWORDS = ("演唱会", "concert", "展览", "conference", "ICRA", "ICLR", "NeurIPS", "票", "ticket")

_CN_NAME_STOPLIST = {
    "小红书", "关注", "分享", "点赞", "收藏", "评论", "图片", "视频",
    "中国", "北京", "上海", "广州", "深圳", "香港", "教育", "设计", "工作", "学习",
}


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Run a battery of cheap regexes to surface candidate entities."""
    if not text:
        return {}

    out: dict[str, list[str]] = {}

    addrs = list({m.group(0) for m in _ADDRESS_RE.finditer(text)})
    if addrs:
        out["address_fragments"] = addrs[:5]

    names = [n for n in _CN_NAME_RE.findall(text) if n not in _CN_NAME_STOPLIST and 2 <= len(n) <= 3]
    if names:
        out["person_names"] = list(dict.fromkeys(names))[:5]

    brands = list({m.group(0) for m in _BRAND_RE.finditer(text)})
    if brands:
        out["brand_names"] = brands[:5]

    models = list({m.group(0) for m in _MODEL_NUM_RE.finditer(text)})
    if models:
        out["model_numbers"] = models[:5]

    low = text.lower()
    if any(kw in low or kw in text for kw in _EDU_KEYWORDS):
        out["edu_keywords"] = [kw for kw in _EDU_KEYWORDS if kw in low or kw in text][:5]
    if any(kw in low or kw in text for kw in _ID_KEYWORDS):
        out["id_keywords"] = [kw for kw in _ID_KEYWORDS if kw in low or kw in text][:5]
    if any(kw in low or kw in text for kw in _NAV_KEYWORDS):
        out["navigation_keywords"] = [kw for kw in _NAV_KEYWORDS if kw in low or kw in text][:5]
    if any(kw in low or kw in text for kw in _EVENT_KEYWORDS):
        out["event_keywords"] = [kw for kw in _EVENT_KEYWORDS if kw in low or kw in text][:5]

    return out


# ── Perception component ──

class Perception:
    """One-shot perception over a single post.

    Args:
      ocr_fn : optional callable (image_path) -> ocr_text. If None, OCR is skipped.
      vl_fn  : optional callable (image_paths, prompt) -> vl_text. If None, VL is skipped.
      vl_prompt_path : optional prompt template path.
    """

    def __init__(
        self,
        *,
        ocr_fn=None,
        vl_fn=None,
        vl_prompt_path: Path | None = None,
    ):
        self.ocr_fn = ocr_fn
        self.vl_fn = vl_fn
        self.vl_prompt = (vl_prompt_path or _default_prompt_path("perception_vl.txt")).read_text(encoding="utf-8")

    def process(
        self,
        post_id: str,
        *,
        caption: str = "",
        media_files: list[str] | None = None,
        timestamp: str | None = None,
    ) -> PerceptualSignature:
        media_files = media_files or []

        # ── Run OCR per image, plus a concatenated post-level view ──
        ocr_text = ""
        ocr_by_image: dict[int, str] = {}
        if self.ocr_fn and media_files:
            try:
                parts = []
                for idx, p in enumerate(media_files, start=1):
                    try:
                        text = self.ocr_fn(p) or ""
                        if text.strip():
                            clean = text.strip()
                            ocr_by_image[idx] = clean
                            parts.append(f"[image {idx}] {clean}")
                    except Exception as exc:
                        logger.warning("OCR failed on %s: %s", p, exc)
                ocr_text = "\n".join(parts)
            except Exception as exc:
                logger.warning("OCR pipeline failed: %s", exc)

        # ── Run VL caption per image ──
        image_summaries: list[dict[str, Any]] = []
        if self.vl_fn and media_files:
            for idx, p in enumerate(media_files, start=1):
                vl_text = self.vl_fn([p], self.vl_prompt) or ""
                tag, caption_text, vl_entities = _parse_vl_block(vl_text)
                image_summaries.append({
                    "image_index": idx,
                    "path": p,
                    "vl_tag": tag,
                    "vl_caption": caption_text,
                    "entities": vl_entities,
                    "ocr_text": ocr_by_image.get(idx, ""),
                })

        tag = _select_post_tag(image_summaries)
        vl_caption = _summarize_image_captions(image_summaries)

        # ── Entity extraction over (caption + OCR + per-image VL captions) ──
        combined = "\n".join(filter(None, [caption, ocr_text, vl_caption]))
        entities = _extract_entities(combined)
        vl_entities = [
            ent
            for item in image_summaries
            for ent in (item.get("entities") or [])
            if ent and str(ent).lower() != "none"
        ]
        if vl_entities:
            entities["vl_entities"] = list(dict.fromkeys(map(str, vl_entities)))[:10]

        return PerceptualSignature(
            post_id=post_id,
            ocr_text=ocr_text,
            vl_caption=vl_caption,
            vl_tag=tag,
            image_summaries=image_summaries,
            entities=entities,
            image_count=len(media_files),
            has_text_in_image=bool(ocr_text.strip()),
            raw_post_text=caption,
        )


# ── Helpers ──

def _parse_vl_block(text: str) -> tuple[str, str, list[str]]:
    """Parse the TAG / CAPTION / ENTITIES block emitted by perception_vl.txt."""
    if not text:
        return "", "", []
    tag = ""
    caption = ""
    entities: list[str] = []
    for line in text.splitlines():
        m_tag = re.match(r"\s*TAG\s*[:：]\s*(.+)", line, re.I)
        m_cap = re.match(r"\s*CAPTION\s*[:：]\s*(.+)", line, re.I)
        m_ent = re.match(r"\s*ENTITIES\s*[:：]\s*(.+)", line, re.I)
        if m_tag and not tag:
            tag = m_tag.group(1).strip().lower()
        elif m_cap and not caption:
            caption = m_cap.group(1).strip()
        elif m_ent and not entities:
            raw_entities = m_ent.group(1).strip()
            if raw_entities.lower() != "none":
                entities = [
                    item.strip()
                    for item in re.split(r"[,，;；、]", raw_entities)
                    if item.strip()
                ]
    if not caption:
        # fallback: first non-empty line
        for line in text.splitlines():
            if line.strip():
                caption = line.strip()
                break
    return tag, caption, entities


_TAG_PRIORITY = {
    "id_card": 0,
    "document": 1,
    "navigation": 2,
    "landmark": 3,
    "signage": 4,
    "hospital": 5,
    "school": 6,
    "graduation": 7,
    "workplace": 8,
    "vehicle": 9,
    "luxury": 10,
    "product": 11,
    "travel": 12,
    "wedding": 13,
    "screenshot": 14,
    "selfie": 15,
    "food_local": 16,
    "scenery": 17,
    "pet": 18,
    "meme": 19,
    "plain": 99,
}


def _select_post_tag(image_summaries: list[dict[str, Any]]) -> str:
    tags = [str(item.get("vl_tag") or "").lower() for item in image_summaries]
    tags = [tag for tag in tags if tag]
    if not tags:
        return ""
    return min(tags, key=lambda tag: _TAG_PRIORITY.get(tag, 50))


def _summarize_image_captions(image_summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in image_summaries:
        idx = item.get("image_index")
        tag = item.get("vl_tag") or "none"
        caption = item.get("vl_caption") or ""
        if caption:
            lines.append(f"Image {idx} [{tag}]: {caption}")
    return "\n".join(lines)


def _default_prompt_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / filename
