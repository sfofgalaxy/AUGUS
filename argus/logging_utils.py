"""Tiny logging helpers for human-readable ARGUS progress output."""
from __future__ import annotations

from datetime import datetime


def argus_log(message: str) -> None:
    """Print a flushed ARGUS progress line with local wall-clock time."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[ARGUS {stamp}] {message}", flush=True)
