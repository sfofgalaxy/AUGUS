"""Path helpers for ARGUS local media inputs."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from argus.config import project_root

IMAGE_MEDIA_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
})


def normalize_media_ref(value: str) -> str:
    """Normalize whitespace around a media reference.

    Open-source inputs should use either an existing absolute path, a path
    relative to the repository/current working directory, an HTTP(S) URL, or a
    data URL. Project-specific path remapping is intentionally not included.
    """
    return value.strip()


def is_image_media_ref(value: str) -> bool:
    """Return True only for media refs that point to image-like formats."""
    value = value.strip()
    if not value:
        return False
    low = value.lower()
    if low.startswith("data:"):
        return low.startswith("data:image/")
    if low.startswith(("http://", "https://")):
        suffix = Path(urlparse(value).path).suffix.lower()
        return suffix in IMAGE_MEDIA_EXTENSIONS
    return Path(value).suffix.lower() in IMAGE_MEDIA_EXTENSIONS


def resolve_local_path(value: str) -> str:
    """Resolve a local relative path while preserving URLs.

    Relative media refs are accepted from several call sites: some are written
    relative to the project root, some relative to the current working
    directory. Return the first existing candidate so downstream image callers
    do not accidentally pass a bare local path as an API image URL.
    """
    value = value.strip()
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        project_root() / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((project_root() / path).resolve())
