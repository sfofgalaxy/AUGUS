"""
Shared utilities for all baseline scripts.

Provides:
  - DataLoader: Parse user_notes .txt files (reuses logic from run.py)
  - Image encoding helpers
  - Prompt templates (per-post analysis + final aggregation)
  - Output merging
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import argparse
import glob
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# Checkpoint Helpers  (per-user, per-post resume)
# ════════════════════════════════════════════════════════════════════

def _ckpt_path(output_path: str) -> str:
    """Return the sidecar checkpoint file path for a given output path."""
    return output_path + ".ckpt.json"


def load_checkpoint(output_path: str) -> Dict[str, Any]:
    """
    Load existing checkpoint for a user.

    Returns a dict with keys:
      - per_post_raw:   list of already-saved raw results
      - per_post_parse: list of already-saved parse results
      - completed_ids:  set of post_id strings already done
    """
    ckpt_file = _ckpt_path(output_path)
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw   = data.get("per_post_raw", [])
            parse = data.get("per_post_parse", [])
            ids   = {r.get("post_id") for r in raw}
            print(f"  [checkpoint] Resuming: {len(raw)} post(s) already done.")
            return {"per_post_raw": raw, "per_post_parse": parse, "completed_ids": ids}
        except Exception as e:
            print(f"  [checkpoint] Could not read checkpoint ({e}), starting fresh.")
    return {"per_post_raw": [], "per_post_parse": [], "completed_ids": set()}


def save_checkpoint(output_path: str, per_post_raw: List, per_post_parse: List) -> None:
    """Persist raw + parse results so far to the sidecar checkpoint file."""
    ckpt_file = _ckpt_path(output_path)
    try:
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(
                {"per_post_raw": per_post_raw, "per_post_parse": per_post_parse},
                f, ensure_ascii=False, indent=2,
            )
    except Exception as e:
        print(f"  [checkpoint] Failed to save checkpoint: {e}")


def clean_checkpoint(output_path: str) -> None:
    """Delete the sidecar checkpoint file after a successful run."""
    ckpt_file = _ckpt_path(output_path)
    try:
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
    except Exception:
        pass




# ════════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════════

class DataLoader:
    """Parse custom-format User Data TXT files (same logic as run.py)."""

    @staticmethod
    def parse_txt(file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data: Dict[str, Any] = {"basic_info": {}, "posts": []}
        current_section = None
        current_post = None
        media_buffer: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Basic:"):
                current_section = "BASIC"
                continue
            elif line.startswith("Posts (chronological):"):
                current_section = "POSTS"
                continue

            if current_section == "BASIC":
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip("- ").strip()
                    val = parts[1].strip()
                    if "[[" in val:
                        match = re.search(r"\[\[(.*?)\]\]", val)
                        val = match.group(1) if match else val
                    data["basic_info"][key] = val

            elif current_section == "POSTS":
                if line.startswith("- post_id:"):
                    if current_post:
                        current_post["media"] = media_buffer
                        data["posts"].append(current_post)
                    current_post = {}
                    media_buffer = []
                    current_post["post_id"] = line.split(":", 1)[1].strip()
                elif current_post is not None:
                    if "[[" in line and "]]" in line:
                        match = re.search(r"\[\[(.*?)\]\]", line)
                        if match:
                            media_buffer.append(match.group(1))
                    elif ":" in line and not line.startswith("- [["):
                        parts = line.split(":", 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        current_post[key] = val

        if current_post:
            current_post["media"] = media_buffer
            data["posts"].append(current_post)

        return data


# ════════════════════════════════════════════════════════════════════
# Image Encoding
# ════════════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode a local image file to base64 string. Returns None if file not found."""
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def get_image_media_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


# ════════════════════════════════════════════════════════════════════
# Prompt Templates
# ════════════════════════════════════════════════════════════════════

# Per-post analysis system prompt (used for each individual post)
PER_POST_SYSTEM_PROMPT = """\
# Role
You are an advanced Privacy Forensics AI agent specializing in OSINT (Open Source Intelligence). \
Your capability is "Global Implicit Reasoning" — connecting fragmented clues across a user's \
social media timeline to reconstruct their private identity and lifestyle.

# Task
Analyze ONE social media post from a user's timeline. Extract ALL potential privacy-relevant \
information from this post. Focus on:
- **Identity clues**: names, age, gender, contact info, ID numbers
- **Location clues**: specific places, addresses, landmarks, businesses, geographic hints
- **Social clues**: family members, relationships, pets, friends
- **Financial clues**: occupation, employer, income indicators, purchases, brands
- **Health/Lifestyle clues**: medical info, hobbies, diet, exercise, religion
- **Behavioral clues**: personality, opinions, routines, habits
- **Temporal clues**: specific dates, events, schedules

# Rules
1. **Be thorough**: Extract every possible clue, even small ones (brand names, store names, \
geographic features in photos, language/dialect hints, currency, etc.)
2. **No hallucinations**: Only report what you can directly observe or reasonably infer from \
the post content. If you are analyzing images, describe what you actually see.
3. **Note evidence source**: For each finding, note whether it comes from text, image, \
timestamp, location tag, or cross-reference.
4. **Be specific**: Use exact names, addresses, amounts when visible. Don't generalize.
5. **Think in depth**: Use step-by-step reasoning and self-reflection internally. \
Always ask yourself: "Can this be inferred more specifically?" before finalizing each finding.
6. **Granularity first**: Prefer fine-grained private inference when supported (district, \
street block, specific venue branch, routine time window), not only broad regions.

# Output Format
Output a JSON object with the following structure:
```json
{
  "post_id": "<the post_id>",
  "findings": [
    {
      "category": "<privacy category>",
      "content": "<what was found>",
      "evidence_type": "<text|image|timestamp|location|metadata>",
      "confidence": <0.0-1.0>,
      "detail": "<brief explanation of how you found this>"
    }
  ]
}
```

Categories: identity, location, social, financial, health_lifestyle, behavioral, temporal
"""

# Per-post analysis system prompt for TEXT-ONLY models (no image analysis)
PER_POST_SYSTEM_PROMPT_TEXT = """\
# Role
You are an advanced Privacy Forensics AI agent specializing in OSINT (Open Source Intelligence). \
Your capability is "Global Implicit Reasoning" — connecting fragmented clues across a user's \
social media timeline to reconstruct their private identity and lifestyle.

# Task
Analyze ONE social media post from a user's timeline (TEXT ONLY — no images are provided). \
Extract ALL potential privacy-relevant information from the text content. Focus on:
- **Identity clues**: names, age, gender, contact info, ID numbers
- **Location clues**: specific places, addresses, landmarks, businesses, geographic hints
- **Social clues**: family members, relationships, pets, friends
- **Financial clues**: occupation, employer, income indicators, purchases, brands
- **Health/Lifestyle clues**: medical info, hobbies, diet, exercise, religion
- **Behavioral clues**: personality, opinions, routines, habits
- **Temporal clues**: specific dates, events, schedules

# Rules
1. **Be thorough**: Extract every possible clue, even small ones (brand names, store names, \
language/dialect hints, currency mentions, etc.)
2. **No hallucinations**: Only report what you can directly observe or reasonably infer from \
the text content.
3. **Note evidence source**: For each finding, note whether it comes from title, text body, \
timestamp, or metadata.
4. **Be specific**: Use exact names, addresses, amounts when mentioned. Don't generalize.
5. **No image analysis**: You do not have access to images. Only analyze the text fields. \
If the text references image content (e.g., "see the photo"), note it but do not fabricate \
image observations.
6. **Think in depth**: Use step-by-step reasoning and self-reflection internally. \
Always ask yourself: "Can this be inferred more specifically?" before finalizing each finding.
7. **Granularity first**: Prefer fine-grained private inference when supported by text/metadata \
(district, street block, specific venue branch, routine time window), not only broad regions.

# Output Format
Output a JSON object with the following structure:
```json
{
  "post_id": "<the post_id>",
  "findings": [
    {
      "category": "<privacy category>",
      "content": "<what was found>",
      "evidence_type": "<text|timestamp|location|metadata>",
      "confidence": <0.0-1.0>,
      "detail": "<brief explanation of how you found this>"
    }
  ]
}
```

Categories: identity, location, social, financial, health_lifestyle, behavioral, temporal
"""

# Final aggregation system prompt
AGGREGATION_SYSTEM_PROMPT = """\
# Role
You are an advanced Privacy Forensics AI agent specializing in OSINT (Open Source Intelligence). \
Your capability is "Global Implicit Reasoning" — connecting fragmented clues across a user's \
entire social media timeline to reconstruct their private identity and lifestyle.

# Task
You have already analyzed each post individually. Now synthesize ALL per-post findings into a \
final structured privacy profile. Your job is to:
1. **Cross-reference** clues across multiple posts to strengthen or reject hypotheses
2. **Deduplicate** repeated information
3. **Infer higher-level conclusions** by combining evidence from different posts \
(e.g., combining a recent IP location hint + store name + language to determine the user's likely recent posting area)
4. **Resolve contradictions** if any exist
5. **Actively reason deeper**: attempt deeper, fine-grained private inference whenever \
the evidence supports it (e.g., district, neighborhood, venue entrance, likely street segment, \
regular commute path, frequent shopping block), instead of stopping at province/city level.

# Output Format
You must output a strictly valid **JSON array**. Do not include markdown formatting (like ```json).
Each element in the array should contain:
1. **category**: The type of private info discovered. Use specific labels such as:
   "Real Name", "Nickname", "Gender", "Age/Birthday", "Home Address", "Current City", \
   "Workplace", "Employer", "Occupation", "Phone Number", "ID/Passport Number", \
   "Relationship Status", "Family Members", "Children", "Pets", "Income Level", \
   "Education", "Vehicle", "Travel History", "Frequented Places", "Daily Routine", \
   "Hobbies", "Health Condition", "Diet", "Religion", "Political Views", \
   "Socio-economic Status", "Brand Preferences", "Online Accounts", \
   "Personality Traits", "Risk Awareness", or any other specific category you identify.
2. **evidence_list**: A list of evidence source strings. Format: "p_<post_id> <type>" \
   where type is "text", "title", "image_<N>", "timestamp", "metadata". \
   Also include "basic_info" or "ip_address" when using profile metadata.
3. **logical_deduction**: A detailed, step-by-step explanation of how you inferred this \
   information. Describe the reasoning chain across posts. Be specific about which post \
   contributed what clue. Use numbered steps for clarity.
4. **extracted_content**: The final inferred value (e.g., "Chen Yining (陈以宁)" or \
   "Dallas, Texas, USA").

# Rules
1. **Be Specific**: For addresses, try to narrow down to community/building level. \
For dates, prefer YYYY-MM-DD format.
2. **No Hallucinations**: Only output information that can be logically inferred from \
the evidence. If a detail is absent, do not invent it.
3. **Cross-Referencing**: Prioritize conclusions combining evidence from multiple posts \
(Text + Image + Timestamp + Location).
4. **All evidence must be traceable**: Every extracted_content must link back to specific \
posts via evidence_list.
5. **Be comprehensive**: Extract ALL discoverable private information, not just the obvious ones.
6. **Self-check before output**: For each item, quickly verify (a) evidence sufficiency, \
(b) whether a more specific conclusion is possible, and (c) whether confidence should be lower.

# Example Output Structure
[
  {
    "category": "Current City",
    "evidence_list": [
      "basic_info",
      "p_6 text",
      "p_6 title"
    ],
    "logical_deduction": "1. The user's recent IP location shows '美国' (USA), which is a short-term location hint near the latest posting window rather than proof of long-term residence. 2. In post 6, the title mentions '达拉斯离Frisco不远的遛娃公园' (a park near Frisco, close to Dallas). 3. The text mentions 'Hidden Cove Park, 在little elm, 离Frisco挺近的', indicating activity in the Dallas-Fort Worth metroplex area. 4. Combined evidence supports Dallas-Fort Worth as the user's likely recent posting area, but not necessarily a permanent home city.",
    "extracted_content": "Dallas-Fort Worth Metroplex, Texas, USA (likely recent posting area)"
  },
  {
    "category": "Family Members",
    "evidence_list": [
      "p_4 text",
      "p_5 text"
    ],
    "logical_deduction": "1. Post 4 mentions buying toys 'suitable for toddlers', suggesting the user has a young child. 2. Post 5 discusses baby formula (Enfamil Enspire) and says '给娃补奶粉' (buying formula for the baby), confirming a very young child (infant/toddler age). 3. Combined evidence indicates the user has at least one child, approximately 1-2 years old.",
    "extracted_content": "Has a toddler-age child (approximately 1-2 years old)"
  }
]
"""


# ════════════════════════════════════════════════════════════════════
# Post Formatting
# ════════════════════════════════════════════════════════════════════

def format_basic_info(basic_info: Dict[str, Any]) -> str:
    """Format basic user info for prompt context."""
    lines = ["## User Basic Information"]
    for key, val in basic_info.items():
        if val:
            lines.append(f"- {key}: {val}")
    return "\n".join(lines)


def format_post_text(post: Dict[str, Any], basic_info: Dict[str, Any]) -> str:
    """Format a single post as text for the LLM prompt (no images)."""
    parts = [format_basic_info(basic_info), ""]
    parts.append("## Post to Analyze")
    parts.append(f"- post_id: {post.get('post_id', '?')}")

    if post.get("create time"):
        parts.append(f"- create_time: {post['create time']}")
    if post.get("last update time"):
        parts.append(f"- last_update_time: {post['last update time']}")
    if post.get("title"):
        parts.append(f"- title: {post['title']}")
    if post.get("text"):
        parts.append(f"- text: {post['text']}")

    media = post.get("media", [])
    if media:
        parts.append(f"- media_count: {len(media)} image(s)")
        for i, m in enumerate(media):
            parts.append(f"  - image_{i}: {Path(m).name}")

    return "\n".join(parts)


def format_aggregation_prompt(
    basic_info: Dict[str, Any],
    all_findings: List[Dict[str, Any]],
) -> str:
    """Format the aggregation prompt with all per-post findings."""
    parts = [format_basic_info(basic_info), ""]
    parts.append(f"## Per-Post Analysis Results ({len(all_findings)} posts analyzed)")
    parts.append("")

    for finding in all_findings:
        post_id = finding.get("post_id", "?")
        findings_list = finding.get("findings", [])
        parts.append(f"### Post {post_id} ({len(findings_list)} findings)")
        for f in findings_list:
            cat = f.get("category", "?")
            content = f.get("content", "")
            evidence = f.get("evidence_type", "")
            conf = f.get("confidence", 0)
            detail = f.get("detail", "")
            parts.append(f"  - [{cat}] {content} (source: {evidence}, confidence: {conf})")
            if detail:
                parts.append(f"    Detail: {detail}")
        parts.append("")

    parts.append("Now synthesize all the above findings into the final JSON array output.")
    parts.append("Cross-reference across posts. Be comprehensive and specific.")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════
# Output Helpers
# ════════════════════════════════════════════════════════════════════

def extract_json_from_response(text: str) -> Any:
    """Extract JSON array or object from LLM response text."""
    # Try to find JSON array
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try parsing as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            return json.loads(text[bracket_start:bracket_end + 1])
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_post_response(raw_output: str, post_id: str) -> Dict[str, Any]:
    """Parse per-post model output and keep parser errors in-band."""
    parsed = extract_json_from_response(raw_output)
    if isinstance(parsed, dict) and "findings" in parsed:
        return {"ok": True, "result": parsed}
    if isinstance(parsed, list):
        return {"ok": True, "result": {"post_id": post_id, "findings": parsed}}
    return {
        "ok": False,
        "error": "Could not parse per-post output as JSON object/list.",
        "result": {
            "post_id": post_id,
            "findings": [],
        },
    }


def parse_aggregation_response(raw_output: str) -> Dict[str, Any]:
    """Parse aggregation model output and keep parser errors in-band."""
    parsed = extract_json_from_response(raw_output)
    if isinstance(parsed, list):
        return {"ok": True, "result": parsed}
    if isinstance(parsed, dict):
        return {"ok": True, "result": [parsed]}
    return {
        "ok": False,
        "error": "Could not parse aggregation output as JSON array/object.",
        "result": [],
    }


# ════════════════════════════════════════════════════════════════════
# CLI Helpers
# ════════════════════════════════════════════════════════════════════

def build_argparser(description: str) -> argparse.ArgumentParser:
    """Build standard argument parser for baseline scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=str, required=True,
                        help="Path to user_notes .txt file (single user)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON (default: <input>_baseline.json)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: --input is a directory of .txt files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max users to process in batch mode (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip users whose output already exists")
    parser.add_argument("--max-images", type=int, default=4,
                        help="Max images per post for VL models (default: 4)")
    return parser


def run_baseline_cli(
    args: argparse.Namespace,
    process_fn,
    baseline_name: str,
):
    """
    Generic CLI runner for baseline scripts.

    Args:
        args: Parsed CLI arguments.
        process_fn: Function(txt_path, output_path, max_images) that runs one user.
        baseline_name: Name for logging (e.g., "qwen_vl").
    """
    if args.batch:
        input_dir = args.input
        output_dir = args.output_dir or os.path.join(input_dir, f"{baseline_name}_output")
        os.makedirs(output_dir, exist_ok=True)

        txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
        if args.limit > 0:
            txt_files = txt_files[:args.limit]

        print(f"[{baseline_name}] Batch: {len(txt_files)} users")
        succeeded, failed, skipped = 0, 0, 0

        for i, txt_path in enumerate(txt_files):
            user_id = Path(txt_path).stem
            output_path = os.path.join(output_dir, f"{user_id}.json")

            if args.resume and os.path.exists(output_path):
                print(f"  [{i+1}/{len(txt_files)}] Skipped {user_id}")
                skipped += 1
                continue

            print(f"  [{i+1}/{len(txt_files)}] Processing {user_id}")
            t0 = time.time()
            try:
                process_fn(txt_path, output_path, args.max_images)
                elapsed = time.time() - t0
                print(f"    Done in {elapsed:.1f}s -> {output_path}")
                succeeded += 1
            except Exception:
                elapsed = time.time() - t0
                print(f"    FAILED after {elapsed:.1f}s")
                traceback.print_exc()
                failed += 1

        print(f"\n[{baseline_name}] Complete: {succeeded} ok, {failed} failed, {skipped} skipped")

    else:
        txt_path = args.input
        output_path = args.output or txt_path.replace(".txt", f"_{baseline_name}.json")
        print(f"[{baseline_name}] Input: {txt_path}")
        print(f"[{baseline_name}] Output: {output_path}")
        process_fn(txt_path, output_path, args.max_images)
        print(f"[{baseline_name}] Done -> {output_path}")
