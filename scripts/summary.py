#!/usr/bin/env python3
"""Print a compact comparison table for one or more rbbench runs."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HEADERS = (
    "Benchmark",
    "Judge score",
    "Input tokens",
    "Cached input tokens",
    "Output tokens",
    "Reasoning tokens",
    "Steps",
    "Duration",
    "Cost",
    "Successful tasks / $",
)
ALIASES = {
    "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens"),
    "cached_input_tokens": (
        "cached_input_tokens",
        "cachedInputTokens",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
    ),
    "output_tokens": (
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "completionTokens",
    ),
    "reasoning_tokens": ("reasoning_tokens", "reasoningTokens"),
    "steps": ("steps", "step_count", "stepCount"),
}


def _load_report_run():
    script = Path(__file__).resolve().parent / "report_run.py"
    spec = importlib.util.spec_from_file_location("report_run", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_run = _load_report_run()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON from {path}: {exc}") from exc


def parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",")]
    if not names or any(not item for item in names):
        raise ValueError("benchmark names must be a comma-separated list without empty names")
    return names


def numeric_metric(metrics: dict[str, Any], metric: str) -> int | None:
    for alias in ALIASES[metric]:
        value = metrics.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def token_usage_fallback(artifact_root: Path) -> dict[str, int]:
    totals: dict[str, int] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "steps": 0,
    }
    found: set[str] = set()
    usage_dir = artifact_root / "tokenUsage"
    for path in sorted(usage_dir.glob("task-*.json")) if usage_dir.is_dir() else []:
        document = read_json(path)
        if not isinstance(document, dict):
            raise ValueError(f"Token usage file is not an object: {path}")
        document_totals = document.get("totals")
        if isinstance(document_totals, dict):
            for metric in totals:
                if metric == "steps":
                    continue
                value = numeric_metric(document_totals, metric)
                if value is not None:
                    totals[metric] += value
                    found.add(metric)
        attempts = document.get("attempts")
        if isinstance(attempts, list):
            executor_steps = 0
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                invocations = attempt.get("invocations")
                if not isinstance(invocations, list):
                    continue
                executor_steps += sum(
                    1
                    for invocation in invocations
                    if isinstance(invocation, dict)
                    and invocation.get("kind") == "executor_step"
                )
            totals["steps"] += executor_steps
            found.add("steps")
    return {metric: value for metric, value in totals.items() if metric in found}


def trajectory_step_fallback(artifact_root: Path) -> int | None:
    path = artifact_root / "steps.jsonl"
    if not path.is_file():
        return None
    count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read trajectory {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
        if isinstance(record, dict) and isinstance(record.get("steps"), list):
            count += len(record["steps"])
    return count


def task_score(result: dict[str, Any]) -> float | None:
    score = result.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    judgement = result.get("judgement")
    if isinstance(judgement, dict) and isinstance(judgement.get("verdict"), bool):
        return 1.0 if judgement["verdict"] else 0.0
    return None


def task_duration_seconds(result: dict[str, Any]) -> float | None:
    """Return a task's end-to-end duration, preferring the runner's measurement."""
    value = result.get("duration_seconds")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        execution = result.get("execution")
        metrics = execution.get("metrics") if isinstance(execution, dict) else None
        value = metrics.get("duration_seconds") if isinstance(metrics, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def metrics_cost(metrics: dict[str, Any]) -> Decimal | None:
    value = metrics.get("cost")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    return None


def format_cost(total_cost: Decimal | None) -> str:
    if total_cost is None:
        return "N/A"
    return f"${float(report_run.round_money(total_cost)):.2f}"


def format_success_per_dollar(successful: int, total_cost: Decimal | None) -> str:
    if total_cost is None or total_cost <= 0:
        return "N/A"
    return f"{float(report_run.round_money(Decimal(successful) / total_cost)):.2f}"


def summarize_run(
    name: str,
    *,
    results_root: Path,
    attempts_root: Path,
    catalog: dict[str, report_run.ModelPrice] | None = None,
    model_maps: dict[str, str] | None = None,
    now: float | None = None,
) -> list[str]:
    run_dir = results_root / name
    if not run_dir.is_dir():
        raise ValueError(f"Benchmark results directory not found: {run_dir}")
    result_paths = sorted(run_dir.glob("RBA-*.json"))
    if not result_paths:
        raise ValueError(f"Benchmark has no task results: {run_dir}")

    scores: list[float] = []
    successful = 0
    aggregate = {metric: 0 for metric in ALIASES}
    observed: set[str] = set()
    task_durations: list[float] = []
    priced_cost = Decimal(0)
    priced_invocations = 0
    metrics_cost_total = Decimal(0)
    metrics_cost_observed = False
    price_catalog = catalog or {}
    maps = model_maps or {}

    for result_path in result_paths:
        result = read_json(result_path)
        if not isinstance(result, dict):
            raise ValueError(f"Task result is not an object: {result_path}")
        score = task_score(result)
        if score is not None:
            scores.append(score)
        judgement = result.get("judgement")
        if isinstance(judgement, dict) and judgement.get("verdict") is True:
            successful += 1
        task_duration = task_duration_seconds(result)
        if task_duration is not None:
            task_durations.append(task_duration)
        attempt_id = result.get("attempt_id")
        attempt_dir = attempts_root / attempt_id if isinstance(attempt_id, str) else None
        artifact_root = (
            attempt_dir / "artifacts" / "browser-agent" if attempt_dir else None
        )
        execution = result.get("execution")
        metrics = execution.get("metrics") if isinstance(execution, dict) else None
        metrics = metrics if isinstance(metrics, dict) else {}
        fallback = token_usage_fallback(artifact_root) if artifact_root else {}
        if "steps" not in fallback and artifact_root:
            trajectory_steps = trajectory_step_fallback(artifact_root)
            if trajectory_steps is not None:
                fallback["steps"] = trajectory_steps

        for metric in aggregate:
            value = numeric_metric(metrics, metric)
            if value is None:
                value = fallback.get(metric)
            if value is not None:
                aggregate[metric] += value
                observed.add(metric)

        if artifact_root and price_catalog:
            cost, priced, _used = report_run.cost_from_token_usage(
                artifact_root, price_catalog, maps
            )
            priced_cost += cost
            priced_invocations += priced

        task_cost = metrics_cost(metrics)
        if task_cost is not None:
            metrics_cost_total += task_cost
            metrics_cost_observed = True

    # Wall-clock duration is distorted by parallelism. Sum end-to-end task
    # durations so results remain comparable across different concurrency.
    duration = round(sum(task_durations)) if len(task_durations) == len(result_paths) else None

    judge_score = f"{statistics.fmean(scores) * 100:.2f}%" if scores else "N/A"

    if priced_invocations > 0:
        total_cost: Decimal | None = priced_cost
    elif metrics_cost_observed:
        total_cost = metrics_cost_total
    else:
        total_cost = None

    def formatted(metric: str) -> str:
        return f"{aggregate[metric]:,}" if metric in observed else "N/A"

    return [
        name,
        judge_score,
        formatted("input_tokens"),
        formatted("cached_input_tokens"),
        formatted("output_tokens"),
        formatted("reasoning_tokens"),
        formatted("steps"),
        f"{duration:,}s" if duration is not None else "N/A",
        format_cost(total_cost),
        format_success_per_dollar(successful, total_cost),
    ]


def render_table(rows: list[list[str]]) -> str:
    all_rows = [list(HEADERS), *rows]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(HEADERS))]

    def render(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            aligned = value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            cells.append(f" {aligned} ")
        return "|" + "|".join(cells) + "|"

    separators = []
    for index, width in enumerate(widths):
        marker = "-" * (width + 1) + ("-" if index == 0 else ":")
        separators.append(marker)
    separator = "|" + "|".join(separators) + "|"
    return "\n".join([render(list(HEADERS)), separator, *(render(row) for row in rows)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", help="one benchmark name or comma-separated names")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / ".runs" / "results"
    )
    parser.add_argument(
        "--attempts-dir", type=Path, default=REPO_ROOT / ".runs" / "attempts"
    )
    parser.add_argument(
        "--model-map",
        action="append",
        default=[],
        metavar="SOURCE=GATEWAY_ID",
        help="map a local model alias to a Vercel AI Gateway model ID",
    )
    args = parser.parse_args()
    try:
        catalog = report_run.fetch_gateway_catalog()
        model_maps = report_run.parse_model_maps(args.model_map)
        rows = [
            summarize_run(
                name,
                results_root=args.results_dir,
                attempts_root=args.attempts_dir,
                catalog=catalog,
                model_maps=model_maps,
            )
            for name in parse_names(args.names)
        ]
        print(render_table(rows))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
