#!/usr/bin/env python3
"""Summarize ARGUS profile/step logs for quick debugging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _status_counts(raw: str) -> dict[str, int]:
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}
    parsed = _extract_json(raw)
    for claim in parsed.get("claim_verifications", []) or []:
        if not isinstance(claim, dict):
            counts["UNKNOWN"] += 1
            continue
        status = str(claim.get("status", "")).upper()
        if status in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"):
            counts[status] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts


def summarize(profile_path: Path) -> dict[str, Any]:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    logs = payload.get("step_logs", []) or []
    rows = []
    total_findings = 0
    total_evidence = 0
    status_total = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}

    for log in logs:
        inv = log.get("investigator", {}) or {}
        ver = log.get("verifier", {}) or {}
        findings = int(inv.get("n_findings", 0) or 0)
        evidence = int(ver.get("n_new_evidence", 0) or 0)
        total_findings += findings
        total_evidence += evidence
        counts = _status_counts(str(ver.get("raw_output", "")))
        for key, value in counts.items():
            status_total[key] += value
        rows.append({
            "post": log.get("post_id"),
            "step": log.get("step_id"),
            "action": ver.get("action"),
            "findings": findings,
            "new_evidence": evidence,
            "supported": counts["SUPPORTED"],
            "unsupported": counts["UNSUPPORTED"],
            "contradicted": counts["CONTRADICTED"],
            "unknown": counts["UNKNOWN"],
        })

    return {
        "profile": str(profile_path),
        "steps": len(logs),
        "findings": total_findings,
        "new_evidence": total_evidence,
        "kept_ratio": round(total_evidence / total_findings, 3) if total_findings else None,
        "claim_status": status_total,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ARGUS run outputs.")
    parser.add_argument("output_dir", help="ARGUS output dir, e.g. outputs/argus/nano_generated_results")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    root = Path(args.output_dir)
    profiles = sorted(root.glob("user_*/profile.json"))
    summaries = [summarize(path) for path in profiles]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    for item in summaries:
        print(f"\n{item['profile']}")
        print(
            f"  steps={item['steps']} findings={item['findings']} "
            f"new_evidence={item['new_evidence']} kept_ratio={item['kept_ratio']}"
        )
        print(f"  claim_status={item['claim_status']}")
        for row in item["rows"]:
            if row["findings"] != row["new_evidence"] or row["unsupported"] or row["contradicted"]:
                print(
                    "  "
                    f"{row['post']} step={row['step']} action={row['action']} "
                    f"findings={row['findings']} evidence={row['new_evidence']} "
                    f"S/U/C/?={row['supported']}/{row['unsupported']}/"
                    f"{row['contradicted']}/{row['unknown']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
