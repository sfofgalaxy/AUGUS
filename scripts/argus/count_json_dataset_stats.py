#!/usr/bin/env python
"""Count user/post/media statistics for unified ARGUS JSON datasets.

Default usage:

    python scripts/argus/count_json_dataset_stats.py

This reads:

    inputs/argus/json/xhs
    inputs/argus/json/ins

Media count is counted from post-level media fields. A video item counts as one
media item, the same as an image.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = {
    "xhs": Path("inputs/argus/json/xhs"),
    "ins": Path("inputs/argus/json/ins"),
}


@dataclass
class UserStats:
    user_id: str
    posts: int
    media: int


@dataclass
class DatasetStats:
    name: str
    path: Path
    users: list[UserStats]

    @property
    def user_count(self) -> int:
        return len(self.users)

    @property
    def post_count(self) -> int:
        return sum(item.posts for item in self.users)

    @property
    def media_count(self) -> int:
        return sum(item.media for item in self.users)

    @property
    def avg_posts_per_user(self) -> float:
        return self.post_count / self.user_count if self.user_count else 0.0

    @property
    def avg_media_per_user(self) -> float:
        return self.media_count / self.user_count if self.user_count else 0.0

    @property
    def avg_media_per_post(self) -> float:
        return self.media_count / self.post_count if self.post_count else 0.0


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.glob("*.json")
        if item.is_file() and item.name != "manifest.json"
    )


def load_user_stats(path: Path) -> UserStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    user = data.get("user") or {}
    user_id = str(
        user.get("user_id")
        or data.get("user_id")
        or user.get("username")
        or path.stem
    )
    posts = data.get("posts") or []
    if not isinstance(posts, list):
        posts = []
    return UserStats(
        user_id=user_id,
        posts=len(posts),
        media=sum(count_post_media(post) for post in posts if isinstance(post, dict)),
    )


def count_post_media(post: dict[str, Any]) -> int:
    """Count post media items, with videos counted as one item.

    Unified ARGUS JSON should usually use ``media_files``. The fallback keys are
    here so older converted files or quick generated inputs can still be
    counted without rewriting them first.
    """
    for key in ("media_files", "media"):
        if key in post:
            return len(_coerce_items(post.get(key)))

    total = 0
    for key in ("images", "image_files", "videos", "video_files"):
        total += len(_coerce_items(post.get(key)))
    return total


def _coerce_items(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, tuple):
        return [item for item in value if item not in (None, "")]
    return [value]


def summarize_dataset(name: str, path: Path) -> DatasetStats:
    files = iter_json_files(path)
    users = [load_user_stats(item) for item in files]
    return DatasetStats(name=name, path=path, users=users)


def print_table(stats: list[DatasetStats]) -> None:
    headers = [
        "dataset",
        "users",
        "posts",
        "media",
        "avg_posts/user",
        "avg_media/user",
        "avg_media/post",
        "path",
    ]
    rows = [
        [
            item.name,
            str(item.user_count),
            str(item.post_count),
            str(item.media_count),
            f"{item.avg_posts_per_user:.2f}",
            f"{item.avg_media_per_user:.2f}",
            f"{item.avg_media_per_post:.2f}",
            str(item.path),
        ]
        for item in stats
    ]
    widths = [
        max(len(row[idx]) for row in [headers, *rows])
        for idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count posts and post-level media items for ARGUS JSON datasets."
    )
    parser.add_argument(
        "--xhs-dir",
        type=Path,
        default=DEFAULT_DATASETS["xhs"],
        help="Unified XHS JSON directory or one JSON file.",
    )
    parser.add_argument(
        "--ins-dir",
        type=Path,
        default=DEFAULT_DATASETS["ins"],
        help="Unified Instagram JSON directory or one JSON file.",
    )
    parser.add_argument(
        "--per-user",
        action="store_true",
        help="Also print per-user post/media counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = [
        summarize_dataset("xhs", args.xhs_dir),
        summarize_dataset("ins", args.ins_dir),
    ]
    print_table(stats)

    if args.per_user:
        for dataset in stats:
            print(f"\n[{dataset.name}] per-user")
            for user in dataset.users:
                print(f"{user.user_id}\tposts={user.posts}\tmedia={user.media}")


if __name__ == "__main__":
    main()
