#!/usr/bin/env python3
"""Rejudge saved attempts without rerunning browser actions or lifecycle hooks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbbench.catalog import load_catalog
from rbbench.io import read_json, write_json
from rbbench.judges import NativeLLMJudge
from rbbench.schema import AttemptDescriptor, ExecutionResult


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-evidence-chars", type=int, default=39_500)
    parser.add_argument("--max-output-tokens", type=int, default=4_000)
    parser.add_argument(
        "--openrouter-provider",
        help="route OpenRouter judge calls to exactly this provider endpoint slug",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--all",
        action="store_true",
        help="rejudge every saved execution, including local placeholder verdicts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --all, skip executions already judged by a non-command provider",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retry malformed or failed judge responses this many times",
    )
    return parser.parse_args()


def _attempt(raw: dict, attempt_dir: Path) -> AttemptDescriptor:
    # Mutable environment adapters persist their lifecycle context at
    # attempt.json, with the descriptor nested under "attempt". Read-only
    # attempts persist the descriptor directly.
    descriptor = raw.get("attempt", raw)
    return AttemptDescriptor(
        attempt_id=str(descriptor["attempt_id"]),
        task_id=str(descriptor["task_id"]),
        start_url=str(descriptor["start_url"]),
        attempt_dir=attempt_dir,
        artifact_dir=Path(descriptor["artifact_dir"]),
        session=dict(descriptor.get("session", {})),
        environment_data=dict(descriptor.get("environment_data", {})),
    )


async def main() -> int:
    args = _arguments()
    catalog = load_catalog()
    run_dir = REPO_ROOT / ".runs" / "results" / args.run_id
    attempt_root = REPO_ROOT / ".runs" / "attempts"
    candidates: list[Path] = []
    for path in sorted(run_dir.glob("RBA-*.json")):
        result = read_json(path)
        error = str(result.get("error") or "")
        if args.all and result.get("execution") is not None:
            if (
                args.resume
                and (result.get("judgement") or {}).get("provider") != "command"
                and result.get("judgement") is not None
            ):
                continue
            candidates.append(path)
        elif (
            result.get("status") == "invalid_environment"
            and result.get("execution") is not None
            and ("judge" in error.lower())
        ):
            candidates.append(path)

    api_key_env = args.api_key_env
    if not api_key_env and (urlparse(args.base_url).hostname or "") == "openrouter.ai":
        api_key_env = "OPENROUTER_API_KEY"
    api_key = os.getenv(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise ValueError(f"Set {api_key_env} for the native LLM judge")
    judge = NativeLLMJudge(
        reference_dir=REPO_ROOT / "references" / "tasks",
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_evidence_chars=args.max_evidence_chars,
        max_output_tokens=args.max_output_tokens,
        text_only=True,
        api_key=api_key,
        request_extra_body=(
            {
                "provider": {
                    "only": [args.openrouter_provider],
                    "allow_fallbacks": False,
                }
            }
            if args.openrouter_provider
            else None
        ),
    )
    gate = asyncio.Semaphore(args.parallel)

    async def rejudge(path: Path) -> str | None:
        async with gate:
            result = read_json(path)
            task = catalog.by_id(str(result["task_id"]))
            attempt_dir = attempt_root / str(result["attempt_id"])
            attempt_path = attempt_dir / "executor-attempt.json"
            if not attempt_path.exists():
                # Native executors persist the same descriptor as attempt.json;
                # the command executor additionally writes executor-attempt.json.
                attempt_path = attempt_dir / "attempt.json"
            attempt = _attempt(read_json(attempt_path), attempt_dir)
            execution = ExecutionResult.from_dict(dict(result["execution"]))
            observation_path = attempt_dir / "trusted-observation.json"
            observation = (
                read_json(observation_path) if observation_path.is_file() else {}
            )
            judgement = None
            last_error: Exception | None = None
            for retry in range(max(0, args.retries) + 1):
                try:
                    judgement = await judge.evaluate(
                        task, attempt, execution, observation
                    )
                    break
                except Exception as exc:  # Keep other tasks independently resumable.
                    last_error = exc
                    if retry < max(0, args.retries):
                        await asyncio.sleep(0.5 * (retry + 1))
            if judgement is None:
                assert last_error is not None
                return f"{path.name}: {type(last_error).__name__}: {last_error}"
            result.update(
                {
                    "status": "success" if judgement.verdict else "agent_failure",
                    "score": 1.0 if judgement.verdict else 0.0,
                    "judgement": judgement.to_dict(),
                    "error": None if judgement.verdict else judgement.failure_reason,
                }
            )
            write_json(path, result)
            write_json(attempt_dir / "result.json", result)
            return None

    outcomes = await asyncio.gather(*(rejudge(path) for path in candidates))
    errors = [error for error in outcomes if error is not None]
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        results = [read_json(path) for path in sorted(run_dir.glob("RBA-*.json"))]
        summary["results"] = results
        summary["status_counts"] = dict(Counter(item["status"] for item in results))
        summary["mean_score"] = (
            sum(float(item.get("score", 0)) for item in results) / len(results)
            if results
            else 0.0
        )
        write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "rejudged": len(candidates) - len(errors),
                "failed": errors,
                "run_id": args.run_id,
            }
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
