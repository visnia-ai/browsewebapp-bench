#!/usr/bin/env python3
"""Replace infrastructure-invalid run records with audited rerun results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbbench.io import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run_id")
    parser.add_argument("rerun_id")
    parser.add_argument("task_ids", nargs="+")
    args = parser.parse_args()

    results_root = REPO_ROOT / ".runs" / "results"
    base_dir = results_root / args.base_run_id
    rerun_dir = results_root / args.rerun_id
    rerun_summary = read_json(rerun_dir / "summary.json")
    replacements: list[dict] = []
    for task_id in args.task_ids:
        original = read_json(base_dir / f"{task_id}.json")
        replacement = read_json(rerun_dir / f"{task_id}.json")
        replacement["rerun_provenance"] = {
            "reason": "original attempt invalidated by browser WebSocket/port failure",
            "original_attempt_id": original["attempt_id"],
            "replacement_run_id": args.rerun_id,
        }
        write_json(base_dir / f"{task_id}.json", replacement)
        replacements.append(replacement)

    summary_path = base_dir / "summary.json"
    summary = read_json(summary_path)
    results = [read_json(path) for path in sorted(base_dir.glob("RBA-*.json"))]
    summary["results"] = results
    summary["status_counts"] = dict(Counter(item["status"] for item in results))
    summary["mean_score"] = sum(float(item["score"]) for item in results) / len(results)
    summary["infra_rerun"] = {
        "run_id": args.rerun_id,
        "task_ids": args.task_ids,
        "duration_seconds": rerun_summary["duration_seconds"],
        "reason": "replace two original browser WebSocket/port failures",
    }
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "base_run_id": args.base_run_id,
                "replacements": [item["task_id"] for item in replacements],
                "status_counts": summary["status_counts"],
                "mean_score": summary["mean_score"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
