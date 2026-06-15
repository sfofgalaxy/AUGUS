#!/usr/bin/env python
"""Single-agent baseline with ARGUS-like tools but no specialist sub-agents.

This baseline gives one GPT tool-calling agent access to ARGUS execution tools
and, when supported by the router model, the original images in the same chat
context. It does not use Perception, AMTR, HDI, CPEG, a separate visual
specialist, or an independent verifier.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from argus.config import load_env
from argus.logging_utils import argus_log
from argus.metrics import finish_run, start_run
from argus.metrics import (
    LLMCallMetric,
    current_post_id,
    estimate_cost_usd,
    record_llm_call,
    usage_from_response,
)
from argus.openrouter import ToolDispatcher
from argus.llm.openrouter import (
    INVESTIGATOR_MODEL,
    PROVIDER_NAME,
    assistant_message_from_openrouter,
    make_client,
    reasoning_extra_body,
)
from argus.llm.openai_compat_response import (
    coerce_chat_completion_response,
    response_preview,
)
from argus.llm.qwen import to_data_url
from argus.path_utils import IMAGE_MEDIA_EXTENSIONS
from argus.retry import retry_call
from argus.tool_registry import init_tool_registry
from argus.tools import get_all_execution_tools
from baselines.common import (
    compact_metadata,
    load_json_inputs,
    parse_json_object,
    remove_stale_error_file,
    safe_name,
    should_skip_existing_profile,
    truncate_text,
)


SINGLE_AGENT_PROMPT = """You are the single-context version of the ARGUS Investigator-Manager.

This is a baseline, not the full ARGUS pipeline. You have the same execution
tools as ARGUS's Investigator, excluding any separate visual specialist, and
you must do the whole user-level privacy leakage analysis in one context.

Available tool families:
- web search: google_search, bing_search, fetch_webpage
- map search: amap_poi_search, google_maps_search
- local OCR: run_ocr
- crop and zoom: crop_image, adaptive_zoom

Important ARGUS-style behavior:
1. Act as a manager. Read captions, user metadata, post metadata, image paths,
   and any evidence you collect. Decide which privacy attributes may be exposed.
2. You are the only agent. If images are attached, inspect them yourself in this
   same context. Do not call or request a separate visual specialist.
3. If you notice text, landmarks, schools, workplaces, maps, documents, stores,
   road signs, or location clues, verify with OCR/search/map/crop/zoom tools.
   When crop_image or adaptive_zoom creates a new crop, that crop will be
   attached back into this same chat context for you to inspect directly.
4. ARGUS does not expose public-URL visual lookup tools because post media are
   local files. Do not ask for image search. Use your own visual reasoning,
   OCR, crop, zoom, web search, and map search instead.
5. Do not over-infer weak lifestyle/personality claims. For example, an object
   appearing once in a photo does not by itself prove a stable user preference.
6. Only include profile claims grounded by text, metadata, visible evidence, or
   actual tool output. If evidence is weak, keep confidence low or omit it.

What is intentionally missing compared with full ARGUS:
- no Qwen perception stage provided ahead of time;
- no AMTR route selection;
- no HDI hypothesis pool;
- no CPEG cross-post evidence graph;
- no independent routing verifier;
- no separate visual specialist;
- no separate deterministic profile projection.

Return strict JSON with this shape:
{
  "user_id": "...",
  "attributes": {
    "attribute.name": {
      "value": "specific inferred value",
      "confidence": 0.0,
      "evidence": [
        {
          "post_id": "...",
          "text": "short evidence chain",
          "source": "text|metadata|image|ocr|web_search|map_search"
        }
      ]
    }
  },
  "summary": "brief natural-language summary"
}
"""


def build_user_prompt(user_id: str, metadata: dict[str, Any], posts: list[dict[str, Any]]) -> str:
    compact_posts = []
    for post in posts:
        compact_posts.append({
            "post_id": post.get("post_id"),
            "timestamp": post.get("timestamp"),
            "location_ip": post.get("location_ip"),
            "caption": truncate_text(post.get("caption") or "", 4000),
            "metadata": compact_metadata(post.get("metadata")),
            "media_files": post.get("media_files") or [],
        })
    return (
        f"{SINGLE_AGENT_PROMPT}\n\n"
        f"USER_ID: {user_id}\n"
        f"USER_METADATA:\n{json.dumps(compact_metadata(metadata), ensure_ascii=False, indent=2)}\n\n"
        f"POSTS:\n{json.dumps(compact_posts, ensure_ascii=False, indent=2)}"
    )


def _tool_names(execution_tools: list[Any]) -> list[str]:
    names = [
        getattr(tool, "name", None) or getattr(tool, "__name__", None)
        for tool in execution_tools
    ]
    return [str(name) for name in names if name]


def _collect_image_attachments(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    for post in posts:
        post_id = str(post.get("post_id") or "")
        for image_idx, path in enumerate(post.get("media_files") or [], start=1):
            attachments.append({
                "post_id": post_id,
                "image_index": image_idx,
                "path": path,
            })
    return attachments


def _build_multimodal_content(
    prompt: str,
    posts: list[dict[str, Any]],
    *,
    max_images: int,
    max_payload_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    attached: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    payload_chars = len(prompt)

    for item in _collect_image_attachments(posts):
        if max_images >= 0 and len(attached) >= max_images:
            skipped.append({**item, "reason": "max_images"})
            continue

        image_url = to_data_url(str(item["path"]))
        if image_url is None:
            skipped.append({**item, "reason": "unreadable"})
            continue

        label = (
            f"IMAGE_ATTACHMENT post_id={item['post_id']} "
            f"image_index={item['image_index']} path={item['path']}"
        )
        added_chars = len(label) + len(image_url)
        if max_payload_chars > 0 and payload_chars + added_chars > max_payload_chars:
            skipped.append({**item, "reason": "payload_limit"})
            continue

        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        attached.append(item)
        payload_chars += added_chars

    if skipped:
        content.append({
            "type": "text",
            "text": "SKIPPED_IMAGE_ATTACHMENTS:\n"
            + json.dumps(skipped, ensure_ascii=False, indent=2),
        })

    return content, {
        "attached": attached,
        "skipped": skipped,
        "tool_attached": [],
        "tool_skipped": [],
        "payload_chars": payload_chars,
    }


def _looks_like_image_ref(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if value.startswith(("http://", "https://", "data:image/")):
        return True
    return Path(value.split("?", 1)[0]).suffix.lower() in IMAGE_MEDIA_EXTENSIONS


def _extract_image_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, str):
        stripped = value.strip().strip('"').strip("'")
        if _looks_like_image_ref(stripped):
            refs.append(stripped)
        return refs
    if isinstance(value, list):
        for item in value:
            refs.extend(_extract_image_refs(item))
        return refs
    if isinstance(value, dict):
        for item in value.values():
            refs.extend(_extract_image_refs(item))
        return refs
    return refs


def _tool_output_image_refs(tool_name: str, tool_output: str) -> list[str]:
    if tool_name not in {"crop_image", "adaptive_zoom"}:
        return []
    text = (tool_output or "").strip()
    if not text or text.startswith("Error"):
        return []
    refs: list[str] = []
    try:
        refs.extend(_extract_image_refs(json.loads(text)))
    except json.JSONDecodeError:
        refs.extend(_extract_image_refs(text))

    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref not in seen:
            deduped.append(ref)
            seen.add(ref)
    return deduped


def _build_tool_image_message(
    *,
    tool_name: str,
    tool_output: str,
    image_meta: dict[str, Any],
    max_payload_chars: int,
) -> dict[str, Any] | None:
    refs = _tool_output_image_refs(tool_name, tool_output)
    if not refs:
        return None

    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"TOOL_IMAGE_ATTACHMENTS from {tool_name}: inspect these generated "
            "crop/zoom images directly before continuing."
        ),
    }]
    attached = 0
    for idx, ref in enumerate(refs, start=1):
        image_url = to_data_url(ref)
        if image_url is None:
            image_meta["tool_skipped"].append({
                "tool": tool_name,
                "path": ref,
                "reason": "unreadable",
            })
            continue
        label = f"TOOL_IMAGE tool={tool_name} image_index={idx} path={ref}"
        added_chars = len(label) + len(image_url)
        if max_payload_chars > 0 and image_meta["payload_chars"] + added_chars > max_payload_chars:
            image_meta["tool_skipped"].append({
                "tool": tool_name,
                "path": ref,
                "reason": "payload_limit",
            })
            continue
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        image_meta["tool_attached"].append({
            "tool": tool_name,
            "path": ref,
        })
        image_meta["payload_chars"] += added_chars
        attached += 1

    return {"role": "user", "content": content} if attached else None


def make_single_context_runner(
    *,
    model: str | None,
    tools: list[Any],
    max_iterations: int,
    max_images: int,
    max_payload_chars: int,
):
    model_name = model or INVESTIGATOR_MODEL
    dispatcher = ToolDispatcher(tools)
    client = make_client()

    def runner(prompt: str, posts: list[dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
        content, image_meta = _build_multimodal_content(
            prompt,
            posts,
            max_images=max_images,
            max_payload_chars=max_payload_chars,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        tool_call_count = 0
        last_text = ""

        argus_log(
            f"single_agent vision attachments={len(image_meta['attached'])} "
            f"skipped={len(image_meta['skipped'])} chars={image_meta['payload_chars']}"
        )

        for iteration in range(1, max_iterations + 1):
            started = time.time()
            argus_log(
                f"llm start provider={PROVIDER_NAME} role=single_agent "
                f"model={model_name} iter={iteration}/{max_iterations}"
            )
            try:
                label = f"provider={PROVIDER_NAME} role=single_agent model={model_name}"
                resp = retry_call(
                    label,
                    lambda: _create_valid_chat_completion(
                        client,
                        label=label,
                        model=model_name,
                        messages=messages,
                        tools=dispatcher.schemas,
                        tool_choice="auto",
                        extra_body=reasoning_extra_body(True),
                    ),
                )
            except Exception as exc:
                record_llm_call(LLMCallMetric(
                    post_id=current_post_id(),
                    role="single_agent",
                    provider=PROVIDER_NAME,
                    model=model_name,
                    elapsed_seconds=time.time() - started,
                    ok=False,
                    error=str(exc),
                ))
                argus_log(
                    f"llm FAILED provider={PROVIDER_NAME} role=single_agent "
                    f"iter={iteration} elapsed={time.time() - started:.1f}s error={exc}"
                )
                raise

            usage = usage_from_response(resp)
            elapsed = time.time() - started
            message = resp.choices[0].message
            content_text = message.content or ""
            tool_calls = message.tool_calls or []
            record_llm_call(LLMCallMetric(
                post_id=current_post_id(),
                role="single_agent",
                provider=PROVIDER_NAME,
                model=model_name,
                elapsed_seconds=elapsed,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost_usd=estimate_cost_usd(
                    model_name,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                ),
                ok=True,
                metadata={
                    "tool_call_count": len(tool_calls),
                    "image_count": len(image_meta["attached"]),
                },
            ))
            argus_log(
                f"llm done provider={PROVIDER_NAME} role=single_agent "
                f"iter={iteration} elapsed={elapsed:.1f}s tokens={usage.get('total_tokens', 0)} "
                f"tool_calls={len(tool_calls)}"
            )

            if content_text:
                last_text = content_text

            messages.append(assistant_message_from_openrouter(message))
            if not tool_calls:
                return content_text or last_text, tool_call_count, image_meta

            tool_image_messages: list[dict[str, Any]] = []
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                argus_log(f"tool start name={name}")
                tool_started = time.time()
                tool_output = dispatcher.execute(name, args)
                argus_log(
                    f"tool done name={name} elapsed={time.time() - tool_started:.1f}s "
                    f"chars={len(tool_output)}"
                )
                tool_call_count += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })
                tool_image_message = _build_tool_image_message(
                    tool_name=name,
                    tool_output=tool_output,
                    image_meta=image_meta,
                    max_payload_chars=max_payload_chars,
                )
                if tool_image_message is not None:
                    tool_image_messages.append(tool_image_message)

            if tool_image_messages:
                messages.extend(tool_image_messages)
                argus_log(
                    "single_agent tool image attachments="
                    f"{len(image_meta['tool_attached'])} "
                    f"tool_skipped={len(image_meta['tool_skipped'])} "
                    f"chars={image_meta['payload_chars']}"
                )

        return last_text, tool_call_count, image_meta

    return runner


def _create_valid_chat_completion(client, *, label: str, **kwargs):
    resp = coerce_chat_completion_response(client.chat.completions.create(**kwargs))
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(
            f"invalid OpenAI-compatible response for {label}: "
            f"missing choices; type={type(resp).__name__}; preview={response_preview(resp)}"
        )
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError(
            f"invalid OpenAI-compatible response for {label}: "
            f"missing choices[0].message; preview={response_preview(resp)}"
        )
    return resp


def run(
    input_path: str,
    output_dir: str,
    *,
    model: str | None,
    max_iterations: int,
    max_images: int,
    max_payload_chars: int,
) -> None:
    load_env()
    init_tool_registry()
    execution_tools = get_all_execution_tools()
    tool_names = _tool_names(execution_tools)
    runner = make_single_context_runner(
        model=model,
        tools=execution_tools,
        max_iterations=max_iterations,
        max_images=max_images,
        max_payload_chars=max_payload_chars,
    )

    root = Path(output_dir)
    print(f"[single_agent] tools={', '.join(tool_names)}")
    for json_path, user_id, metadata, posts in load_json_inputs(input_path):
        user_dir = root / safe_name(user_id)
        out_path = user_dir / "profile.json"
        if should_skip_existing_profile(out_path, log_prefix="single_agent", user_id=user_id):
            continue

        print(
            f"[single_agent] user={user_id} posts={len(posts)} "
            f"model={model or 'default'} max_iterations={max_iterations}"
        )
        start_run(user_id)
        raw, tool_call_count, image_meta = runner(build_user_prompt(user_id, metadata, posts), posts)
        metrics = finish_run()
        parsed = parse_json_object(raw)
        if not parsed:
            parsed = {"user_id": user_id, "attributes": {}, "summary": ""}
        parsed.setdefault("user_id", user_id)
        parsed.setdefault("attributes", {})

        payload = {
            "user_id": user_id,
            "baseline": "single_agent",
            "model": model or os.environ.get("ARGUS_INVESTIGATOR_MODEL", "default"),
            "source_json": str(json_path),
            "profile": parsed,
            "raw_output": raw,
            "tool_call_count": tool_call_count,
            "tools": tool_names,
            "image_attachments": image_meta,
            "metrics": metrics.to_dict() if metrics else {},
        }
        user_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        remove_stale_error_file(user_dir)
        print(f"[single_agent] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-context single-agent baseline.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", default="outputs/baselines/single_agent")
    parser.add_argument(
        "--model",
        default=os.environ.get("BASELINE_MODEL"),
        help="GPT backbone model. Defaults to ARGUS_INVESTIGATOR_MODEL / gpt-5.5.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.environ.get("BASELINE_SINGLE_AGENT_MAX_ITER", "12")),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=int(os.environ.get("BASELINE_SINGLE_AGENT_MAX_IMAGES", "-1")),
        help="Maximum user images attached to the same GPT context. -1 means no count limit.",
    )
    parser.add_argument(
        "--max-payload-chars",
        type=int,
        default=int(os.environ.get("BASELINE_SINGLE_AGENT_MAX_PAYLOAD_CHARS", "24000000")),
        help="Approximate prompt+base64 image payload cap. <=0 disables the cap.",
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        model=args.model,
        max_iterations=args.max_iterations,
        max_images=args.max_images,
        max_payload_chars=args.max_payload_chars,
    )


if __name__ == "__main__":
    main()
