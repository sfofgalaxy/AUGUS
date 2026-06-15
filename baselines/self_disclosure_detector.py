#!/usr/bin/env python
"""Self-disclosure detector + user-level aggregation baseline.

Adapted from "Exploring and Detecting Self-disclosure in Multi-modal posts on
Chinese Social Media". The paper uses image descriptions + post text to detect
15 personal-information fields. This implementation keeps that shape, but uses
one selected multimodal model for both image description and disclosure
detection so it stays a single-model baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from argus.config import load_env
from argus.metrics import finish_post, finish_run, start_post, start_run
from baselines.common import (
    chat_baseline_vision,
    chat_baseline_text,
    compact_metadata,
    default_multimodal_model,
    load_json_inputs,
    normalize_model_provider,
    parse_json_object,
    remove_stale_error_file,
    safe_name,
    should_skip_existing_profile,
    truncate_text,
)


DISCLOSURE_FIELDS = [
    "Location",
    "Name",
    "Age",
    "Gender",
    "Marital Status",
    "Pet",
    "Husband/Boyfriend",
    "Wife/Girlfriend",
    "Sexual Orientation",
    "Physical Health",
    "Family Status",
    "Occupation",
    "Mental Health",
    "Education Information",
    "Financial Status",
]

FIELD_TO_ATTR = {
    "Location": "location.self_disclosed",
    "Name": "identity.name",
    "Age": "identity.age",
    "Gender": "identity.gender",
    "Marital Status": "relationship.marital_status",
    "Pet": "social.pet",
    "Husband/Boyfriend": "relationship.male_partner",
    "Wife/Girlfriend": "relationship.female_partner",
    "Sexual Orientation": "identity.sexual_orientation",
    "Physical Health": "health.physical",
    "Family Status": "family.status",
    "Occupation": "work.occupation",
    "Mental Health": "health.mental",
    "Education Information": "education.information",
    "Financial Status": "financial.status",
}

UNKNOWN_VALUES = {"", "unknown", "unk", "n/a", "none", "null", "not mentioned", "cannot determine", "unknown."}


IMAGE_DESCRIPTION_PROMPT = """You are an image describer for self-disclosure detection.

Describe only factual visual content that may help identify personal
information. Transcribe visible text exactly when possible. Avoid speculation,
summarization, or subjective interpretation.

Return concise JSON:
{
  "images": [
    {
      "image_index": 1,
      "description": "factual description",
      "visible_text": "exact visible text or Unknown",
      "privacy_clues": ["location sign", "medical document", "..."]
    }
  ]
}
"""


DETECTOR_PROMPT = """You are an information annotator for multimodal social-media self-disclosure.

Task:
Given one post's text, post metadata, user metadata, and optional image
descriptions, identify the poster's disclosed personal information.

Use this 15-field taxonomy:
- Location: place names, addresses, coordinates, administrative divisions,
  landmarks, transport routes, or other geographic details.
- Name: real personal names.
- Age: exact age, age range, life stage, school-age term, birth date/year.
- Gender: physiological gender or gender identity.
- Marital Status: single, married, divorced, widowed, or unknown.
- Pet: pet ownership and pet type.
- Husband/Boyfriend: relationship with a male partner.
- Wife/Girlfriend: relationship with a female partner.
- Sexual Orientation: heterosexual, homosexual, bisexual, asexual, other.
- Physical Health: physical health status or specific physical conditions.
- Family Status: family members, family structure, family relations, family
  economic status, or family occupation.
- Occupation: job title, student, unemployment, or work type.
- Mental Health: mental health status or specific mental-health conditions.
- Education Information: education level, school status, study abroad, degree.
- Financial Status: income, assets, debts, spending, savings, economic class.

Rules:
- Prioritize the post text. Image descriptions are auxiliary evidence and may
  be wrong.
- If text and image descriptions conflict, follow the text.
- Do not infer stable traits from weak one-off objects.
- If a field is missing or cannot be determined, return value "Unknown".
- Return all 15 fields. No markdown.

Return strict JSON:
{
  "disclosures": {
    "Location": {"value": "Unknown", "evidence": "", "source": "text|image|metadata|both|none", "confidence": 0.0},
    ...
  }
}
"""


AGGREGATE_PROMPT = """You aggregate post-level self-disclosure detections into a user profile.

Use only non-Unknown post-level disclosures. Merge duplicates, keep multiple
plausible values when needed, and preserve evidence post IDs. Do not add new
claims that were not detected at the post level.

Return strict JSON:
{
  "user_id": "...",
  "attributes": {
    "attribute.name": {
      "value": "best value or list of values",
      "confidence": 0.0,
      "evidence": [{"post_id": "...", "field": "...", "text": "...", "source": "..."}]
    }
  },
  "summary": "brief summary"
}
"""


def is_unknown(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in UNKNOWN_VALUES or text in {"未知", "不详", "无法确定", "未提及"}


def vl_describe_post(
    media_files: list[str],
    *,
    model_provider: str,
    model: str,
    max_images: int,
) -> str:
    if not media_files or max_images == 0:
        return ""
    selected = media_files if max_images < 0 else media_files[:max_images]
    if not selected:
        return ""
    return chat_baseline_vision(
        selected,
        IMAGE_DESCRIPTION_PROMPT,
        provider=model_provider,
        model=model,
        role="baseline_self_disclosure_image_description",
        max_images=len(selected),
        temperature=0.0,
    )


def analyze_post(
    post: dict[str, Any],
    *,
    user_metadata: dict[str, Any],
    model_provider: str,
    model: str,
    max_images: int,
) -> dict[str, Any]:
    post_id = str(post.get("post_id") or "unknown_post")
    media_files = list(post.get("media_files") or [])
    start_post(post_id, image_count=len(media_files))
    image_description = vl_describe_post(
        media_files,
        model_provider=model_provider,
        model=model,
        max_images=max_images,
    )
    payload = {
        "user_metadata": compact_metadata(user_metadata),
        "post": {
            "post_id": post_id,
            "timestamp": post.get("timestamp"),
            "location_ip": post.get("location_ip"),
            "caption": truncate_text(post.get("caption") or "", 8000),
            "metadata": compact_metadata(post.get("metadata")),
            "image_count": len(media_files),
            "image_description": image_description,
        },
    }
    raw = chat_baseline_text(
        [
            {"role": "system", "content": DETECTOR_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        provider=model_provider,
        model=model,
        role="baseline_self_disclosure_detect",
        temperature=0.0,
    )
    finish_post(post_id, step_count=1, tool_call_count=0)

    parsed = normalize_detection(parse_json_object(raw))
    return {
        "post_id": post_id,
        "image_description": image_description,
        "raw": raw,
        "parse": parsed,
    }


def normalize_detection(parsed: dict[str, Any]) -> dict[str, Any]:
    disclosures = parsed.get("disclosures") if isinstance(parsed, dict) else None
    if not isinstance(disclosures, dict):
        disclosures = parsed if isinstance(parsed, dict) else {}

    normalized: dict[str, Any] = {}
    for field in DISCLOSURE_FIELDS:
        item = disclosures.get(field)
        if isinstance(item, dict):
            value = item.get("value", "Unknown")
            evidence = item.get("evidence", "")
            source = item.get("source", "none")
            confidence = item.get("confidence", 0.0)
        else:
            value = item if item is not None else "Unknown"
            evidence = ""
            source = "none" if is_unknown(value) else "text"
            confidence = 0.0 if is_unknown(value) else 0.5
        normalized[field] = {
            "value": value if not is_unknown(value) else "Unknown",
            "evidence": str(evidence or ""),
            "source": str(source or "none"),
            "confidence": float(confidence or 0.0),
            "attribute": FIELD_TO_ATTR[field],
        }
    return {"disclosures": normalized}


def aggregate_user(
    user_id: str,
    metadata: dict[str, Any],
    per_post: list[dict[str, Any]],
    *,
    model_provider: str,
    model: str,
) -> dict[str, Any]:
    compact = []
    for item in per_post:
        post_id = item.get("post_id")
        for field, info in (item.get("parse", {}).get("disclosures") or {}).items():
            value = info.get("value")
            if is_unknown(value):
                continue
            compact.append({
                "post_id": post_id,
                "field": field,
                "attribute": info.get("attribute") or FIELD_TO_ATTR.get(field, field),
                "value": value,
                "evidence": truncate_text(info.get("evidence") or "", 600),
                "source": info.get("source") or "unknown",
                "confidence": info.get("confidence") or 0.0,
            })

    prompt = (
        f"{AGGREGATE_PROMPT}\n\n"
        f"USER_ID: {user_id}\n"
        f"USER_METADATA:\n{json.dumps(compact_metadata(metadata), ensure_ascii=False, indent=2)}\n\n"
        f"POST_LEVEL_DISCLOSURES:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )
    raw = chat_baseline_text(
        [{"role": "user", "content": prompt}],
        provider=model_provider,
        model=model,
        role="baseline_self_disclosure_aggregate",
        temperature=0.0,
    )
    parsed = parse_json_object(raw)
    if not parsed:
        parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
    parsed.setdefault("user_id", user_id)
    parsed.setdefault("attributes", {})
    return {"raw": raw, "parse": parsed}


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[self_disclosure_detector] checkpoint unreadable={path} error={exc}")
        return []
    per_post = data.get("per_post")
    return per_post if isinstance(per_post, list) else []


def save_checkpoint(path: Path, per_post: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"per_post": per_post}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def run(
    input_path: str,
    output_dir: str,
    *,
    model_provider: str,
    model: str | None,
    max_images: int,
) -> None:
    load_env()
    model_provider = normalize_model_provider(model_provider)
    model_name = model or default_multimodal_model(model_provider)
    root = Path(output_dir)

    for json_path, user_id, metadata, posts in load_json_inputs(input_path):
        user_dir = root / safe_name(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        out_path = user_dir / "profile.json"
        checkpoint_path = user_dir / "profile.ckpt.json"
        if should_skip_existing_profile(
            out_path,
            log_prefix="self_disclosure_detector",
            user_id=user_id,
        ):
            continue

        print(
            f"[self_disclosure_detector] user={user_id} posts={len(posts)} "
            f"model={model_provider}/{model_name}"
        )
        per_post = load_checkpoint(checkpoint_path)
        done = {str(item.get("post_id")) for item in per_post}
        start_run(user_id)
        for idx, post in enumerate(posts, start=1):
            post_id = str(post.get("post_id") or f"post_{idx:03d}")
            if post_id in done:
                print(f"[self_disclosure_detector] skip post={post_id} checkpoint")
                continue
            result = analyze_post(
                post,
                user_metadata=metadata,
                model_provider=model_provider,
                model=model_name,
                max_images=max_images,
            )
            per_post.append(result)
            done.add(post_id)
            save_checkpoint(checkpoint_path, per_post)

        aggregate = aggregate_user(
            user_id,
            metadata,
            per_post,
            model_provider=model_provider,
            model=model_name,
        )
        metrics = finish_run()
        payload = {
            "user_id": user_id,
            "baseline": "self_disclosure_detector",
            "model_provider": model_provider,
            "model": model_name,
            "source_json": str(json_path),
            "profile": aggregate["parse"],
            "raw": {"per_post": per_post, "aggregation": aggregate["raw"]},
            "metrics": metrics.to_dict() if metrics else {},
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        remove_stale_error_file(user_dir)
        print(f"[self_disclosure_detector] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-disclosure detector baseline.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", default="outputs/baselines/self_disclosure_detector")
    parser.add_argument(
        "--model-provider",
        default=os.environ.get("BASELINE_MODEL_PROVIDER", "gpt"),
        choices=["gpt", "qwen", "gemini", "shadowapi", "dashscope", "vertex_gemini"],
    )
    parser.add_argument("--model", default=os.environ.get("BASELINE_MODEL"))
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("BASELINE_MAX_IMAGES", "4")))
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        model_provider=args.model_provider,
        model=args.model,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
