"""Local runtime metrics for ARGUS runs.

Records wall-clock time, token usage, estimated cost, and per-post summaries.
No external tracing service is used.
"""
from __future__ import annotations

import contextvars
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMCallMetric:
    post_id: str | None
    role: str
    provider: str
    model: str
    elapsed_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostMetric:
    post_id: str
    image_count: int
    started_at: float
    elapsed_seconds: float = 0.0
    investigated: bool = False
    step_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_call_count: int = 0


@dataclass
class RunMetrics:
    user_id: str
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    posts: dict[str, PostMetric] = field(default_factory=dict)
    llm_calls: list[LLMCallMetric] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "reasoning_tokens": self.total_reasoning_tokens,
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": round(self.total_estimated_cost_usd, 8),
                "llm_call_count": len(self.llm_calls),
                "post_count": len(self.posts),
            },
            "posts": {
                post_id: {
                    **asdict(metric),
                    "estimated_cost_usd": round(metric.estimated_cost_usd, 8),
                }
                for post_id, metric in self.posts.items()
            },
            "llm_calls": [
                {
                    **asdict(call),
                    "estimated_cost_usd": round(call.estimated_cost_usd, 8),
                }
                for call in self.llm_calls
            ],
        }


_current_run: contextvars.ContextVar[RunMetrics | None] = contextvars.ContextVar(
    "argus_current_run_metrics",
    default=None,
)
_current_post_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "argus_current_post_id",
    default=None,
)


def start_run(user_id: str) -> RunMetrics:
    metrics = RunMetrics(user_id=user_id)
    _current_run.set(metrics)
    _current_post_id.set(None)
    return metrics


def finish_run() -> RunMetrics | None:
    metrics = _current_run.get()
    if metrics is not None:
        metrics.elapsed_seconds = time.time() - metrics.started_at
    _current_post_id.set(None)
    return metrics


def current_run() -> RunMetrics | None:
    return _current_run.get()


def set_current_post(post_id: str | None) -> None:
    _current_post_id.set(post_id)


def current_post_id() -> str | None:
    return _current_post_id.get()


def start_post(post_id: str, image_count: int) -> None:
    metrics = current_run()
    if metrics is None:
        return
    metrics.posts[post_id] = PostMetric(
        post_id=post_id,
        image_count=image_count,
        started_at=time.time(),
    )
    set_current_post(post_id)


def finish_post(
    post_id: str,
    *,
    step_count: int,
    tool_call_count: int,
) -> None:
    metrics = current_run()
    if metrics is None or post_id not in metrics.posts:
        set_current_post(None)
        return
    post = metrics.posts[post_id]
    post.elapsed_seconds = time.time() - post.started_at
    post.step_count = step_count
    post.tool_call_count = tool_call_count
    post.investigated = step_count > 0
    set_current_post(None)


def record_llm_call(call: LLMCallMetric) -> None:
    metrics = current_run()
    if metrics is None:
        return
    metrics.llm_calls.append(call)
    metrics.total_input_tokens += call.input_tokens
    metrics.total_output_tokens += call.output_tokens
    metrics.total_reasoning_tokens += call.reasoning_tokens
    metrics.total_tokens += call.total_tokens
    metrics.total_estimated_cost_usd += call.estimated_cost_usd

    if call.post_id and call.post_id in metrics.posts:
        post = metrics.posts[call.post_id]
        post.input_tokens += call.input_tokens
        post.output_tokens += call.output_tokens
        post.reasoning_tokens += call.reasoning_tokens
        post.total_tokens += call.total_tokens
        post.estimated_cost_usd += call.estimated_cost_usd
        post.llm_call_count += 1


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    get = usage.get if isinstance(usage, dict) else lambda k, default=0: getattr(usage, k, default)
    input_tokens = int(get("prompt_tokens", get("input_tokens", 0)) or 0)
    output_tokens = int(get("completion_tokens", get("output_tokens", 0)) or 0)
    total_tokens = int(get("total_tokens", input_tokens + output_tokens) or 0)
    reasoning_tokens = 0
    details = get("completion_tokens_details", None)
    if details is not None:
        if isinstance(details, dict):
            reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        else:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def usage_from_gemini_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "total_tokens": total_tokens,
    }


# Rough defaults, USD per 1M tokens. Override by editing this table if your
# provider price changes; metrics are intended for local accounting, not billing.
MODEL_PRICES_USD_PER_MILLION = {
    "anthropic/claude-opus-4.7": {"input": 15.0, "output": 75.0},
    "gpt-5.5": {"input": 0.0, "output": 0.0},
    "qwen3.6-plus": {"input": 0.4, "output": 2.4},
    "qwen3.7-max": {"input": 1.2, "output": 6.0},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = MODEL_PRICES_USD_PER_MILLION.get(model)
    if not price:
        return 0.0
    return (
        input_tokens * price["input"] / 1_000_000
        + output_tokens * price["output"] / 1_000_000
    )
