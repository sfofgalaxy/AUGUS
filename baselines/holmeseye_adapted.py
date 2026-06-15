#!/usr/bin/env python
"""HolmesEye-adapted baseline for ARGUS user JSON inputs.

Adapted from "The Eye of Sherlock Holmes". The original method avoids feeding
all images into one direct VLM call. It uses:
  1. intra-image extraction,
  2. inter-image analysis over small image groups,
  3. LLM summarization with missing-question generation,
  4. focused VLM inquiry,
  5. LLM final decision.

This is a prior-work-style baseline, not ARGUS: no HDI, no CPEG, no routing
verifier, and no external search/map/OCR tool loop. In this adapted version,
the VLM and LLM stages are both executed by the same selected multimodal model
so GPT-vs-Qwen runs remain clean single-model comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from argus.config import load_env
from argus.metrics import finish_run, start_run
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


TARGET_ATTRIBUTES = [
    "identity.name",
    "identity.age",
    "identity.gender",
    "location.home",
    "location.work",
    "location.activity_place",
    "education.institution",
    "education.level",
    "work.occupation",
    "work.company",
    "health.physical",
    "health.mental",
    "relationship.status",
    "family.status",
    "social.pet",
    "financial.status",
    "financial.spending_level",
    "interest.hobby",
    "routine.daily_routine",
    "behavioral.personality",
    "behavioral.social_activity",
    "behavioral.leadership",
]


EXTRACTION_PROMPT = """You are the VLM extraction stage of a HolmesEye-style privacy profiling baseline.

Analyze exactly the provided image(s), but do not produce the final profile.
Extract factual, privacy-relevant visual clues:
- visible people: approximate age/gender only if visually supported;
- locations: landmarks, signs, addresses, transport routes, maps, store names;
- documents/screens: transcribe visible text;
- work/education: uniforms, badges, classrooms, workplaces, certificates;
- health/family/social/financial/lifestyle: only factual visual cues;
- uncertainty: say when a cue is weak or ambiguous.

Return concise JSON:
{
  "observations": [
    {"image_id": "...", "cue": "...", "supports": ["attribute.name"], "confidence": 0.0}
  ],
  "notes": "short factual summary"
}
"""


INTER_IMAGE_PROMPT = """You are the VLM inter-image analysis stage of a HolmesEye-style baseline.

These images belong to the same user. Analyze relationships across the images,
not just each image in isolation:
- repeated locations, signs, objects, clothing, settings, routines;
- style similarities/differences that suggest occupation, education, income,
  daily routine, health, social activity, hobbies, or personality;
- contradictions or uncertainty.

Do not finalize the profile. Return concise JSON:
{
  "cross_image_patterns": [
    {"pattern": "...", "supports": ["attribute.name"], "image_ids": ["..."], "confidence": 0.0}
  ],
  "notes": "short summary"
}
"""


SUMMARIZATION_PROMPT = """You are the LLM summarization stage of a HolmesEye-style privacy profiling baseline.

Given user metadata, post captions, intra-image extraction, and inter-image
analysis, build an initial profile over the target attributes. Also identify
which attributes remain unclear and what focused visual questions should be
asked next.

Rules:
- Use only supplied evidence.
- Keep weak/ambiguous claims low confidence.
- Do not invent facts.
- Generate at most 3 focused missing questions.
- For each question, include up to 8 image_ids that are most relevant. If no
  specific images are known, leave image_ids empty.

Return strict JSON:
{
  "initial_profile": {
    "attribute.name": {
      "value": "...",
      "confidence": 0.0,
      "evidence": [{"image_id": "...", "post_id": "...", "text": "..."}],
      "status": "supported|weak|unknown|contradicted"
    }
  },
  "missing_questions": [
    {"attribute": "attribute.name", "question": "...", "image_ids": ["..."]}
  ],
  "summary": "brief summary"
}
"""


INQUIRY_PROMPT_TEMPLATE = """You are the VLM inquiry stage of a HolmesEye-style baseline.

Focused question:
{question}

Only answer this question from the provided images. If the images do not support
an answer, say unsupported. Return concise JSON:
{{
  "question": "{question_json}",
  "answers": [
    {{"image_id": "...", "answer": "...", "confidence": 0.0, "evidence": "..."}}
  ],
  "conclusion": "supported answer or unsupported"
}}
"""


DECISION_PROMPT = """You are the LLM final decision stage of a HolmesEye-style privacy profiling baseline.

Consolidate the initial profile and focused inquiry evidence into a final user
profile. Use the target attributes when possible, but you may include another
privacy attribute if it is clearly supported.

Rules:
- Use only supplied evidence.
- Prefer specific but supported values.
- Drop unsupported attributes instead of filling the profile with guesses.
- Preserve evidence chains with image_id/post_id when available.
- Return strict JSON only.

Return:
{
  "user_id": "...",
  "attributes": {
    "attribute.name": {
      "value": "specific inferred value",
      "confidence": 0.0,
      "evidence": [{"post_id": "...", "image_id": "...", "text": "...", "source": "intra|inter|inquiry|caption|metadata"}]
    }
  },
  "summary": "brief summary"
}
"""


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def call_vlm(
    image_paths: list[str],
    prompt: str,
    *,
    model_provider: str,
    model: str,
    role: str,
) -> str:
    if not image_paths:
        return ""
    return chat_baseline_vision(
        image_paths,
        prompt,
        provider=model_provider,
        model=model,
        role=role,
        max_images=len(image_paths),
        temperature=0.0,
    )


def collect_images(
    posts: list[dict[str, Any]],
    *,
    max_images_per_post: int,
    max_images_per_user: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for post_idx, post in enumerate(posts, start=1):
        post_id = str(post.get("post_id") or f"post_{post_idx:03d}")
        media_files = list(post.get("media_files") or [])
        if max_images_per_post > 0:
            media_files = media_files[:max_images_per_post]
        for image_idx, path in enumerate(media_files, start=1):
            records.append({
                "image_id": f"{post_id}:img_{image_idx:03d}",
                "post_id": post_id,
                "image_index": image_idx,
                "path": path,
                "caption": truncate_text(post.get("caption") or "", 1000),
                "timestamp": post.get("timestamp"),
                "location_ip": post.get("location_ip"),
                "post_metadata": compact_metadata(post.get("metadata")),
            })
            if max_images_per_user > 0 and len(records) >= max_images_per_user:
                return records
    return records


def compact_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for post in posts:
        compact.append({
            "post_id": post.get("post_id"),
            "timestamp": post.get("timestamp"),
            "location_ip": post.get("location_ip"),
            "caption": truncate_text(post.get("caption") or "", 1200),
            "metadata": compact_metadata(post.get("metadata")),
            "image_count": len(post.get("media_files") or []),
        })
    return compact


def extraction_prompt_for(record: dict[str, Any]) -> str:
    context = {
        "image_id": record["image_id"],
        "post_id": record["post_id"],
        "caption": record.get("caption"),
        "timestamp": record.get("timestamp"),
        "location_ip": record.get("location_ip"),
        "target_attributes": TARGET_ATTRIBUTES,
    }
    return f"{EXTRACTION_PROMPT}\n\nIMAGE_CONTEXT:\n{json.dumps(context, ensure_ascii=False, indent=2)}"


def inter_prompt_for(group: list[dict[str, Any]]) -> str:
    context = [
        {
            "image_id": item["image_id"],
            "post_id": item["post_id"],
            "caption": item.get("caption"),
            "timestamp": item.get("timestamp"),
            "location_ip": item.get("location_ip"),
        }
        for item in group
    ]
    return (
        f"{INTER_IMAGE_PROMPT}\n\n"
        f"TARGET_ATTRIBUTES:\n{json.dumps(TARGET_ATTRIBUTES, ensure_ascii=False, indent=2)}\n\n"
        f"IMAGE_GROUP_CONTEXT:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def parse_missing_questions(initial: dict[str, Any]) -> list[dict[str, Any]]:
    questions = initial.get("missing_questions") if isinstance(initial, dict) else []
    if not isinstance(questions, list):
        return []
    out: list[dict[str, Any]] = []
    for item in questions:
        if isinstance(item, str):
            out.append({"attribute": "unknown", "question": item, "image_ids": []})
        elif isinstance(item, dict) and item.get("question"):
            image_ids = item.get("image_ids")
            out.append({
                "attribute": str(item.get("attribute") or "unknown"),
                "question": str(item.get("question")),
                "image_ids": [str(x) for x in image_ids] if isinstance(image_ids, list) else [],
            })
    return out


def run_user_holmeseye(
    *,
    user_id: str,
    metadata: dict[str, Any],
    posts: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    model_provider: str,
    model: str,
    group_size: int,
    max_inquiry_questions: int,
    inquiry_max_images: int,
) -> dict[str, Any]:
    image_by_id = {item["image_id"]: item for item in image_records}
    extraction = checkpoint.setdefault("intra_image_extraction", {})
    inter = checkpoint.setdefault("inter_image_analysis", {})

    for record in image_records:
        image_id = record["image_id"]
        if image_id in extraction:
            continue
        raw = call_vlm(
            [record["path"]],
            extraction_prompt_for(record),
            model_provider=model_provider,
            model=model,
            role="baseline_holmeseye_extract",
        )
        extraction[image_id] = {
            "image_id": image_id,
            "post_id": record["post_id"],
            "path": record["path"],
            "raw": raw,
            "parse": parse_json_object(raw),
        }
        save_checkpoint(checkpoint_path, checkpoint)

    image_groups = chunks(image_records, group_size)
    for group_idx, group in enumerate(image_groups, start=1):
        group_key = f"group_{group_idx:03d}"
        if group_key in inter:
            continue
        raw = call_vlm(
            [item["path"] for item in group],
            inter_prompt_for(group),
            model_provider=model_provider,
            model=model,
            role="baseline_holmeseye_inter_image",
        )
        inter[group_key] = {
            "group_id": group_key,
            "image_ids": [item["image_id"] for item in group],
            "raw": raw,
            "parse": parse_json_object(raw),
        }
        save_checkpoint(checkpoint_path, checkpoint)

    if "initial_summary" not in checkpoint:
        summary_payload = {
            "user_id": user_id,
            "user_metadata": compact_metadata(metadata),
            "posts": compact_posts(posts),
            "target_attributes": TARGET_ATTRIBUTES,
            "intra_image_extraction": list(extraction.values()),
            "inter_image_analysis": list(inter.values()),
        }
        raw = chat_baseline_text(
            [
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str)},
            ],
            provider=model_provider,
            model=model,
            role="baseline_holmeseye_summarize",
            temperature=0.0,
        )
        checkpoint["initial_summary"] = {"raw": raw, "parse": parse_json_object(raw)}
        save_checkpoint(checkpoint_path, checkpoint)

    if "inquiries" not in checkpoint:
        initial_parse = checkpoint.get("initial_summary", {}).get("parse") or {}
        questions = parse_missing_questions(initial_parse)[:max_inquiry_questions]
        inquiries: list[dict[str, Any]] = []
        for q_idx, question in enumerate(questions, start=1):
            requested_ids = [image_id for image_id in question.get("image_ids", []) if image_id in image_by_id]
            selected = [image_by_id[image_id] for image_id in requested_ids]
            if not selected:
                selected = image_records[:inquiry_max_images]
            if inquiry_max_images > 0:
                selected = selected[:inquiry_max_images]
            grouped_answers = []
            for group_idx, group in enumerate(chunks(selected, group_size), start=1):
                prompt = INQUIRY_PROMPT_TEMPLATE.format(
                    question=question["question"],
                    question_json=json.dumps(question["question"], ensure_ascii=False)[1:-1],
                )
                raw = call_vlm(
                    [item["path"] for item in group],
                    prompt + "\n\nIMAGE_IDS:\n" + json.dumps([item["image_id"] for item in group], ensure_ascii=False),
                    model_provider=model_provider,
                    model=model,
                    role="baseline_holmeseye_inquiry",
                )
                grouped_answers.append({
                    "group": group_idx,
                    "image_ids": [item["image_id"] for item in group],
                    "raw": raw,
                    "parse": parse_json_object(raw),
                })
            inquiries.append({
                "question_id": f"q_{q_idx:03d}",
                "attribute": question["attribute"],
                "question": question["question"],
                "answers": grouped_answers,
            })
            checkpoint["inquiries"] = inquiries
            save_checkpoint(checkpoint_path, checkpoint)

    decision_payload = {
        "user_id": user_id,
        "target_attributes": TARGET_ATTRIBUTES,
        "user_metadata": compact_metadata(metadata),
        "posts": compact_posts(posts),
        "initial_summary": checkpoint.get("initial_summary"),
        "inquiries": checkpoint.get("inquiries", []),
    }
    raw = chat_baseline_text(
        [
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": json.dumps(decision_payload, ensure_ascii=False, indent=2, default=str)},
        ],
        provider=model_provider,
        model=model,
        role="baseline_holmeseye_decision",
        temperature=0.0,
    )
    parsed = parse_json_object(raw)
    if not parsed:
        parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
    parsed.setdefault("user_id", user_id)
    parsed.setdefault("attributes", {})
    checkpoint["final_decision"] = {"raw": raw, "parse": parsed}
    save_checkpoint(checkpoint_path, checkpoint)
    return parsed


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[holmeseye_adapted] checkpoint unreadable={path} error={exc}")
        return {}
    return data if isinstance(data, dict) else {}


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def run(
    input_path: str,
    output_dir: str,
    *,
    model_provider: str,
    model: str | None,
    group_size: int,
    max_images_per_post: int,
    max_images_per_user: int,
    max_inquiry_questions: int,
    inquiry_max_images: int,
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
        if should_skip_existing_profile(out_path, log_prefix="holmeseye_adapted", user_id=user_id):
            continue

        image_records = collect_images(
            posts,
            max_images_per_post=max_images_per_post,
            max_images_per_user=max_images_per_user,
        )
        print(
            f"[holmeseye_adapted] user={user_id} posts={len(posts)} images={len(image_records)} "
            f"model={model_provider}/{model_name}"
        )

        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint.setdefault("image_records", image_records)
        start_run(user_id)
        profile = run_user_holmeseye(
            user_id=user_id,
            metadata=metadata,
            posts=posts,
            image_records=image_records,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            model_provider=model_provider,
            model=model_name,
            group_size=group_size,
            max_inquiry_questions=max_inquiry_questions,
            inquiry_max_images=inquiry_max_images,
        )
        metrics = finish_run()
        payload = {
            "user_id": user_id,
            "baseline": "holmeseye_adapted",
            "model_provider": model_provider,
            "model": model_name,
            "group_size": group_size,
            "source_json": str(json_path),
            "profile": profile,
            "raw": checkpoint,
            "metrics": metrics.to_dict() if metrics else {},
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        remove_stale_error_file(user_dir)
        print(f"[holmeseye_adapted] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HolmesEye-adapted baseline.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", default="outputs/baselines/holmeseye_adapted")
    parser.add_argument(
        "--model-provider",
        default=os.environ.get("BASELINE_MODEL_PROVIDER", "gpt"),
        choices=["gpt", "qwen", "gemini", "shadowapi", "dashscope", "vertex_gemini"],
    )
    parser.add_argument("--model", default=os.environ.get("BASELINE_MODEL"))
    parser.add_argument("--group-size", type=int, default=int(os.environ.get("HOLMESEYE_GROUP_SIZE", "3")))
    parser.add_argument(
        "--max-images-per-post",
        type=int,
        default=int(os.environ.get("HOLMESEYE_MAX_IMAGES_PER_POST", "0")),
        help="0 means no per-post cap.",
    )
    parser.add_argument(
        "--max-images-per-user",
        type=int,
        default=int(os.environ.get("HOLMESEYE_MAX_IMAGES_PER_USER", "0")),
        help="0 means no per-user cap.",
    )
    parser.add_argument(
        "--max-inquiry-questions",
        type=int,
        default=int(os.environ.get("HOLMESEYE_MAX_INQUIRY_QUESTIONS", "3")),
    )
    parser.add_argument(
        "--inquiry-max-images",
        type=int,
        default=int(os.environ.get("HOLMESEYE_INQUIRY_MAX_IMAGES", "8")),
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        model_provider=args.model_provider,
        model=args.model,
        group_size=args.group_size,
        max_images_per_post=args.max_images_per_post,
        max_images_per_user=args.max_images_per_user,
        max_inquiry_questions=args.max_inquiry_questions,
        inquiry_max_images=args.inquiry_max_images,
    )


if __name__ == "__main__":
    main()
