"""Run the full ARGUS pipeline on unified user JSON inputs."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from argus.runner import build_argus_app
from baselines.common import (
    load_json_inputs,
    remove_stale_error_file,
    safe_name,
    should_skip_existing_profile,
    write_json_result,
)


def _error_message(*, user_id: str, error: Exception) -> str:
    return f"[[ERROR]] stage=argus_json_user user={user_id} reason={type(error).__name__}: {error}"


def _write_error_file(
    *,
    user_dir: Path,
    user_id: str,
    source_json: Path,
    error: Exception,
) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    error_path = user_dir / "error.json"
    payload: dict[str, Any] = {
        "user_id": user_id,
        "pipeline": "argus",
        "source_json": str(source_json),
        "message": _error_message(user_id=user_id, error=error),
        "traceback": traceback.format_exc(),
    }
    error_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return error_path


def run(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = True,
    limit: int = 0,
) -> None:
    app = build_argus_app()
    root = Path(output_dir)
    inputs = load_json_inputs(input_path)
    if limit > 0:
        inputs = inputs[:limit]

    root.mkdir(parents=True, exist_ok=True)
    print(f"[argus_json] users={len(inputs)} input={input_path} output={root}")

    succeeded = skipped = 0
    for idx, (json_path, user_id, metadata, posts) in enumerate(inputs, start=1):
        user_dir = root / safe_name(user_id)
        out_path = user_dir / "profile.json"
        if resume and should_skip_existing_profile(out_path, log_prefix="argus_json", user_id=user_id):
            skipped += 1
            continue

        run_metadata = dict(metadata)
        run_metadata["source_json"] = str(json_path)
        print(f"[argus_json] [{idx}/{len(inputs)}] start user={user_id} posts={len(posts)}")
        started = time.time()
        try:
            result = app(user_id, run_metadata, posts)
            profile_path = write_json_result(result, user_dir)
            remove_stale_error_file(user_dir)
            print(
                f"[argus_json] [{idx}/{len(inputs)}] done user={user_id} "
                f"elapsed={time.time() - started:.1f}s -> {profile_path}"
            )
            succeeded += 1
        except Exception as exc:
            error_path = _write_error_file(
                user_dir=user_dir,
                user_id=user_id,
                source_json=json_path,
                error=exc,
            )
            print(
                f"[argus_json] [{idx}/{len(inputs)}] FAILED user={user_id} "
                f"elapsed={time.time() - started:.1f}s error={error_path}"
            )
            traceback.print_exc()
            if os.environ.get("ARGUS_CONTINUE_ON_ERROR", "0") != "1":
                raise

    print("=" * 60)
    print(f"[argus_json] complete succeeded={succeeded} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ARGUS on unified user JSON inputs.")
    parser.add_argument("--input", required=True, help="One user JSON file or a directory of JSON files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for ARGUS profiles.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip successful existing profiles.")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N users.")
    args = parser.parse_args()
    run(
        args.input,
        args.output_dir,
        resume=not args.no_resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
