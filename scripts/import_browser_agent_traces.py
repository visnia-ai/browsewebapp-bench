#!/usr/bin/env python3
"""Import standalone Browser Agent traces into a rejudgeable rbbench run.

The source directory is read-only from this script's perspective: trajectories
and token-usage files are copied into attempt artifacts and are never moved or
modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbbench.browser_agent_artifacts import convert_trajectory, load_token_usage, load_trajectory
from rbbench.io import write_json
from rbbench.schema import ExecutionResult


TRACE_PATTERN = re.compile(r"steps-task-(\d+)\.jsonl$")
WEBSITE_PATTERN = re.compile(r"(?:^|\\n)website:\s*(https?://\S+)", re.IGNORECASE)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", help="run id (defaults to the source directory name)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add only missing trajectories to an existing imported run",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / ".runs" / "results")
    parser.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / ".runs" / "attempts")
    return parser.parse_args()


def _start_url(task: str) -> str:
    match = WEBSITE_PATTERN.search(task)
    return match.group(1).rstrip(".,") if match else "https://example.invalid/"


def _catalog_task(task_id: str, task: str, start_url: str) -> dict:
    return {
        "task_id": task_id,
        "title": f"Imported Browser Agent task {task_id.removeprefix('RBA-BU100-')}",
        "confirmed_task": task,
        "category": "imported/public_web",
        "environment": {
            "adapter": "public_web",
            "kind": "public_web",
            "start_url": start_url,
            "auth": "unknown",
            "mutable": False,
        },
        "fixture": {},
        "oracle": {
            "type": "llm_semantic",
            "assertions": [
                {
                    "kind": "semantic_completion",
                    "description": "Judge all material requirements in the imported task.",
                }
            ],
        },
        "cleanup": {"strategy": "none", "verify_absence": False},
        "safety": {"forbidden_actions": [], "external_side_effects": False},
        "sources": [start_url],
        "tags": ["imported", "browser-agent", "bu100"],
        "release_status": "candidate",
    }


def main() -> int:
    args = _arguments()
    source = args.source.resolve()
    if not source.is_dir():
        raise ValueError(f"Trace directory does not exist: {source}")
    run_id = args.name or source.name
    run_dir = args.results_dir / run_id
    run_existed = run_dir.exists()
    if run_existed and not args.append:
        raise ValueError(f"Destination run already exists: {run_dir}")
    if args.append and not run_existed:
        raise ValueError(f"Cannot append because destination run does not exist: {run_dir}")

    traces: list[tuple[int, Path]] = []
    for path in source.glob("steps-task-*.jsonl"):
        match = TRACE_PATTERN.match(path.name)
        if match:
            traces.append((int(match.group(1)), path))
    if not traces:
        raise ValueError(f"No steps-task-NNN.jsonl traces found in {source}")

    run_dir.mkdir(parents=True, exist_ok=args.append)
    if args.append:
        summary_path = run_dir / "summary.json"
        catalog_path = run_dir / "catalog.json"
        if not summary_path.is_file() or not catalog_path.is_file():
            raise ValueError(f"Existing run is not an imported trace run: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        results = list(summary.get("results", []))
        catalog_tasks = list(catalog.get("tasks", []))
    else:
        results = []
        catalog_tasks = []
    existing_ids = {str(item.get("task_id")) for item in results}
    imported_count = 0
    created_attempts: list[Path] = []
    created_results: list[Path] = []
    try:
        for number, trace_path in sorted(traces):
            suffix = f"{number:03d}"
            task_id = f"RBA-BU100-{suffix}"
            if task_id in existing_ids:
                continue
            attempt_id = f"{run_id}-bu100-{suffix}"
            attempt_dir = args.runtime_dir / attempt_id
            if attempt_dir.exists():
                raise ValueError(f"Destination attempt already exists: {attempt_dir}")
            artifact_root = attempt_dir / "artifacts" / "browser-agent"
            artifact_root.mkdir(parents=True)
            created_attempts.append(attempt_dir)

            artifact = load_trajectory(trace_path)
            task = artifact.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Trace has no task text: {trace_path}")
            usage_source = source / "tokenUsage" / f"task-{suffix}.json"
            usage = load_token_usage(usage_source)
            projected = convert_trajectory(artifact, expected_task=task, usage_totals=usage)

            copied_trace = artifact_root / "steps.jsonl"
            copied_usage = artifact_root / "tokenUsage" / "task-001.json"
            copied_usage.parent.mkdir()
            shutil.copy2(trace_path, copied_trace)
            shutil.copy2(usage_source, copied_usage)

            start_url = _start_url(task)
            descriptor = {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "start_url": start_url,
                "attempt_dir": str(attempt_dir.resolve()),
                "artifact_dir": str((attempt_dir / "artifacts").resolve()),
                "session": {},
                "environment_data": {
                    "import_source": str(source),
                    "source_trace": str(trace_path),
                    "source_task_number": number,
                },
            }
            execution = ExecutionResult(
                final_result=projected.final_result,
                steps=projected.steps,
                screenshots=[],
                observation={"result": {"final_result": projected.final_result}},
                metrics={
                    "steps": projected.num_steps,
                    "duration_seconds": projected.duration_seconds,
                    "input_tokens": projected.input_tokens,
                    "cached_input_tokens": projected.cached_input_tokens,
                    "output_tokens": projected.output_tokens,
                    "reasoning_tokens": projected.reasoning_tokens,
                    "non_reasoning_output_tokens": projected.non_reasoning_output_tokens,
                    "total_tokens": projected.total_tokens,
                    "model_invocations": projected.model_invocations,
                    "postprocessor": "imported-browser-agent-bu-projection-v1",
                    "source_trajectory": str(trace_path),
                },
            )
            result = {
                "task_id": task_id,
                "attempt_id": attempt_id,
                "status": "invalid_environment",
                "score": 0.0,
                "judgement": None,
                "execution": asdict(execution),
                "error": "Imported trace pending judge",
                "cleanup_error": None,
                "duration_seconds": projected.duration_seconds,
            }
            write_json(attempt_dir / "attempt.json", descriptor)
            write_json(attempt_dir / "task.json", _catalog_task(task_id, task, start_url))
            write_json(attempt_dir / "trusted-observation.json", {})
            write_json(attempt_dir / "result.json", result)
            result_path = run_dir / f"{task_id}.json"
            write_json(result_path, result)
            created_results.append(result_path)
            results.append(result)
            catalog_tasks.append(_catalog_task(task_id, task, start_url))
            existing_ids.add(task_id)
            imported_count += 1

        results.sort(key=lambda item: str(item["task_id"]))
        catalog_tasks.sort(key=lambda item: str(item["task_id"]))
        catalog = {
            "name": f"Imported {run_id}",
            "version": "1",
            "description": f"Generated from {source}; no reference ground truth supplied.",
            "tasks": catalog_tasks,
        }
        write_json(run_dir / "catalog.json", catalog)
        write_json(
            run_dir / "import-manifest.json",
            {
                "run_id": run_id,
                "source": str(source),
                "source_files_modified": False,
                "imported": len(results),
                "imported_this_invocation": imported_count,
            },
        )
        write_json(
            run_dir / "summary.json",
            {
                "run_id": run_id,
                "results": results,
                "status_counts": dict(Counter(item["status"] for item in results)),
                "mean_score": (
                    sum(float(item.get("score", 0)) for item in results) / len(results)
                    if results
                    else 0.0
                ),
            },
        )
    except Exception:
        if not run_existed:
            shutil.rmtree(run_dir, ignore_errors=True)
        for attempt_dir in created_attempts:
            shutil.rmtree(attempt_dir, ignore_errors=True)
        for result_path in created_results:
            result_path.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "run_id": run_id,
                "imported": imported_count,
                "total": len(results),
                "run_dir": str(run_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
