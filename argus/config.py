"""Small ARGUS runtime configuration helpers.

ARGUS keeps its default path independent from the legacy pydantic settings
module. This loader is intentionally tiny: it reads `.env` into `os.environ`
when present and resolves the project root used by skill-backed tools.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from `.env` without overriding env vars."""
    path = Path(env_path) if env_path is not None else project_root(from_env=False) / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _clean_env_value(value.strip())


def project_root(*, from_env: bool = True) -> Path:
    if from_env:
        configured = os.environ.get("PROJECT_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _clean_env_value(value: str) -> str:
    value = _strip_inline_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in ("'", '"'):
            quote = None if quote == ch else ch if quote is None else quote
            continue
        if ch == "#" and quote is None and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx]
    return value

