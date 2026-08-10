#!/usr/bin/env python3
"""Recover judgeable executions when Browser Agent's internal verifier rejects."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbbench.browser_agent_artifacts import (
    convert_trajectory,
    load_token_usage,
    load_trajectory,
)
from rbbench.io import read_json, write_json
from rbbench.schema import ExecutionResult


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    result_dir = REPO_ROOT / ".runs" / "results" / args.run_id
    attempt_root = REPO_ROOT / ".runs" / "attempts"
    recovered: list[str] = []
    for result_path in sorted(result_dir.glob("RBA-*.json")):
        result = read_json(result_path)
        if result.get("execution") is not None:
            continue
        attempt_dir = attempt_root / str(result["attempt_id"])
        attempt = read_json(attempt_dir / "attempt.json")
        # Mutable environment adapters nest the descriptor under "attempt".
        descriptor = attempt.get("attempt", attempt)
        artifact_root = Path(descriptor["artifact_dir"]) / "browser-agent"
        trajectory_path = artifact_root / "steps.jsonl"
        usage_path = artifact_root / "tokenUsage" / "task-001.json"
        if not trajectory_path.is_file() or not usage_path.is_file():
            continue
        trajectory = load_trajectory(trajectory_path)
        task = trajectory.get("task")
        if not isinstance(task, str):
            continue
        projected = convert_trajectory(
            trajectory,
            expected_task=task,
            usage_totals=load_token_usage(usage_path),
        )
        screenshots = sorted(
            str(path)
            for path in Path(descriptor["artifact_dir"]).rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        execution = ExecutionResult(
            final_result=projected.final_result,
            steps=projected.steps,
            screenshots=screenshots,
            observation={
                "result": {"final_result": projected.final_result},
                "safety": {},
                "page": {"url": descriptor["start_url"]},
            },
            metrics={
                "steps": projected.num_steps,
                "duration_seconds": projected.duration_seconds,
                "cost": 0.0,
                "input_tokens": projected.input_tokens,
                "cached_input_tokens": projected.cached_input_tokens,
                "output_tokens": projected.output_tokens,
                "reasoning_tokens": projected.reasoning_tokens,
                "non_reasoning_output_tokens": projected.non_reasoning_output_tokens,
                "total_tokens": projected.total_tokens,
                "model_invocations": projected.model_invocations,
                "postprocessor": "recovered-browser-agent-bu-projection-v1",
                "source_trajectory": str(trajectory_path),
            },
        )
        result["execution"] = asdict(execution)
        write_json(result_path, result)
        write_json(attempt_dir / "result.json", result)
        recovered.append(str(result["task_id"]))
    print(json.dumps({"run_id": args.run_id, "recovered": recovered}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
