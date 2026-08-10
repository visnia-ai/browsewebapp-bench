#!/usr/bin/env python3
"""Verify that the native judge rejects incomplete and plausible-wrong runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbbench.catalog import load_catalog
from rbbench.io import write_json
from rbbench.judges import NativeLLMJudge
from rbbench.schema import AttemptDescriptor, ExecutionResult


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="comma-separated task ids")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".runs" / "negative-judge-checks.json",
    )
    return parser.parse_args()


async def main() -> int:
    args = arguments()
    catalog = load_catalog()
    task_ids = [value.strip() for value in args.task.split(",") if value.strip()]
    tasks = [catalog.by_id(task_id) for task_id in task_ids]
    judge = NativeLLMJudge(
        reference_dir=REPO_ROOT / "references" / "tasks",
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_evidence_chars=39_500,
        max_output_tokens=4_000,
        text_only=True,
    )
    gate = asyncio.Semaphore(args.parallel)
    results: list[dict[str, object]] = []

    async def check(task, variant: str, execution: ExecutionResult) -> None:
        async with gate:
            with tempfile.TemporaryDirectory(prefix="rbbench-negative-judge-") as raw:
                root = Path(raw)
                attempt = AttemptDescriptor(
                    attempt_id=f"negative-{variant}-{task.task_id.lower()}",
                    task_id=task.task_id,
                    start_url=task.environment.start_url,
                    attempt_dir=root,
                    artifact_dir=root,
                )
                judgement = await judge.evaluate(task, attempt, execution, {})
                results.append(
                    {
                        "task_id": task.task_id,
                        "variant": variant,
                        "passed": judgement.verdict is False,
                        "judgement": judgement.to_dict(),
                    }
                )

    jobs = []
    for task in tasks:
        jobs.append(check(task, "incomplete", ExecutionResult()))
        jobs.append(
            check(
                task,
                "wrong_result",
                ExecutionResult(
                    final_result=(
                        "Completed the requested browser workflow. The terminal result "
                        "was exactly 999999 and every requested record had status UNKNOWN."
                    ),
                    steps=[
                        f"Opened {task.environment.start_url} in the browser UI.",
                        "Navigated to a plausible result page and returned the displayed values.",
                    ],
                ),
            )
        )
    await asyncio.gather(*jobs)
    results.sort(key=lambda item: (str(item["task_id"]), str(item["variant"])))
    report = {
        "passed": all(bool(item["passed"]) for item in results),
        "task_count": len(tasks),
        "check_count": len(results),
        "results": results,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
