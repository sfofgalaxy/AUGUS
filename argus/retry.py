"""Small retry helpers for ARGUS external calls."""
from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

from argus.logging_utils import argus_log

T = TypeVar("T")


def max_retries() -> int:
    raw = os.environ.get("ARGUS_LLM_MAX_RETRIES", "5")
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def gemini_max_retries() -> int:
    # 7 attempts means 6 wait intervals: 15, 30, 60, 120, 240, 480 by default.
    raw = os.environ.get("ARGUS_GEMINI_MAX_RETRIES", "7")
    try:
        return max(1, int(raw))
    except ValueError:
        return 7


def base_interval_seconds() -> float:
    raw = os.environ.get("ARGUS_LLM_RETRY_BASE_INTERVAL", "15")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 15.0


def retry_call(label: str, fn: Callable[[], T], *, attempts: int | None = None) -> T:
    total = attempts or _default_attempts_for_label(label)
    base_interval = base_interval_seconds()
    last_exc: Exception | None = None
    for attempt in range(1, total + 1):
        try:
            if attempt > 1:
                argus_log(f"retry {label} attempt={attempt}/{total}")
            return fn()
        except Exception as exc:
            last_exc = exc
            argus_log(f"call failed {label} attempt={attempt}/{total} error={exc}")
            if _is_non_retryable(exc, label=label):
                argus_log(f"not retrying non-retryable {label}")
                raise
            if attempt < total:
                wait_seconds = base_interval * (2 ** (attempt - 1))
                argus_log(f"retry wait {label} seconds={wait_seconds:g}")
                time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


def _default_attempts_for_label(label: str) -> int:
    if "vertex_gemini" in label or "gemini" in label.lower():
        return gemini_max_retries()
    return max_retries()


def _is_non_retryable(exc: Exception, *, label: str = "") -> bool:
    text = str(exc)
    if "data_inspection_failed" in text or "inappropriate content" in text:
        return "provider=dashscope" in label

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 425, 429}:
            return False
        return 400 <= status_code < 500

    markers = (
        "invalid_request_error",
        "InvalidParameter",
        "String value length",
        "maximum allowed",
    )
    return any(marker in text for marker in markers)
