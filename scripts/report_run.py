#!/usr/bin/env python3
"""Generate a YAML performance and cost report for an rbbench run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"
PERCENTILES = tuple(range(0, 101, 5))
GRAPH_WIDTH = 30


class TwoDecimalFloat(float):
    """Marker for YAML floats that must retain two decimal places."""


class ReportDumper(yaml.SafeDumper):
    pass


def _represent_two_decimal_float(
    dumper: yaml.SafeDumper, value: TwoDecimalFloat
) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:float", f"{float(value):.2f}"
    )


ReportDumper.add_representer(TwoDecimalFloat, _represent_two_decimal_float)


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


ReportDumper.add_representer(str, _represent_string)


def round_int(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def round_money(value: Decimal | float) -> TwoDecimalFloat:
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return TwoDecimalFloat(rounded)


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def percentile_graph(values: list[float]) -> str:
    points = [(item, round_int(percentile(values, item))) for item in PERCENTILES]
    maximum = max((value for _, value in points), default=0)
    lines: list[str] = []
    for item, value in points:
        width = round_int(value / maximum * GRAPH_WIDTH) if maximum > 0 else 0
        bar = "#" * width
        label = f"p{item}"
        lines.append(f"{label:>4} | {bar:<{GRAPH_WIDTH}} {value}")
    return "\n".join(lines)


def distribution(values: Iterable[float], *, include_count: bool = False) -> dict[str, Any]:
    collected = list(values)
    result: dict[str, Any] = {}
    if include_count:
        result["count"] = len(collected)
    result["median"] = round_int(statistics.median(collected)) if collected else 0
    result["mean"] = round_int(statistics.fmean(collected)) if collected else 0
    result["percentilesGraph"] = percentile_graph(collected)
    return result


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON from {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read trajectory {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"Trajectory record in {path} is not an object")
        records.append(record)
    return records


@dataclass(frozen=True)
class PriceTier:
    minimum: int
    maximum: int | None
    cost_per_token: Decimal

    def applies(self, token_count: int) -> bool:
        return token_count >= self.minimum and (
            self.maximum is None or token_count < self.maximum
        )


@dataclass(frozen=True)
class ModelPrice:
    gateway_id: str
    input_per_token: Decimal
    cached_input_per_token: Decimal | None
    cache_write_per_token: Decimal | None
    output_per_token: Decimal
    input_tiers: tuple[PriceTier, ...] = ()
    cached_input_tiers: tuple[PriceTier, ...] = ()
    cache_write_tiers: tuple[PriceTier, ...] = ()
    output_tiers: tuple[PriceTier, ...] = ()

    @staticmethod
    def _tiered_rate(
        base: Decimal, tiers: tuple[PriceTier, ...], input_tokens: int
    ) -> Decimal:
        for tier in tiers:
            if tier.applies(input_tokens):
                return tier.cost_per_token
        return base

    def input_rate(self, input_tokens: int) -> Decimal:
        return self._tiered_rate(self.input_per_token, self.input_tiers, input_tokens)

    def cached_input_rate(self, input_tokens: int) -> Decimal:
        base = (
            self.cached_input_per_token
            if self.cached_input_per_token is not None
            else self.input_per_token
        )
        return self._tiered_rate(base, self.cached_input_tiers, input_tokens)

    def cache_write_rate(self, input_tokens: int) -> Decimal:
        base = (
            self.cache_write_per_token
            if self.cache_write_per_token is not None
            else self.input_per_token
        )
        return self._tiered_rate(base, self.cache_write_tiers, input_tokens)

    def output_rate(self, input_tokens: int) -> Decimal:
        return self._tiered_rate(self.output_per_token, self.output_tiers, input_tokens)

    def report_value(self) -> dict[str, float | None]:
        million = Decimal(1_000_000)
        return {
            "inputPerM": float(self.input_per_token * million),
            "cachedInputPerM": (
                float(self.cached_input_per_token * million)
                if self.cached_input_per_token is not None
                else None
            ),
            "cacheWriteInputPerM": (
                float(self.cache_write_per_token * million)
                if self.cache_write_per_token is not None
                else None
            ),
            "outputPerM": float(self.output_per_token * million),
        }


def fetch_gateway_catalog(timeout_seconds: float = 15.0) -> dict[str, ModelPrice]:
    request = urllib.request.Request(
        GATEWAY_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "rbbench-report/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to fetch Vercel AI Gateway prices: {exc}") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("Vercel AI Gateway returned an invalid model catalog")
    catalog: dict[str, ModelPrice] = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        pricing = model.get("pricing")
        if not isinstance(pricing, dict):
            continue
        try:
            input_rate = Decimal(str(pricing["input"]))
            output_rate = Decimal(str(pricing["output"]))
        except (KeyError, ArithmeticError):
            continue
        cached_raw = pricing.get("input_cache_read")
        cached_rate = Decimal(str(cached_raw)) if cached_raw is not None else None
        cache_write_raw = pricing.get("input_cache_write")
        cache_write_rate = (
            Decimal(str(cache_write_raw)) if cache_write_raw is not None else None
        )
        gateway_id = model["id"]
        catalog[gateway_id] = ModelPrice(
            gateway_id=gateway_id,
            input_per_token=input_rate,
            cached_input_per_token=cached_rate,
            cache_write_per_token=cache_write_rate,
            output_per_token=output_rate,
            input_tiers=_parse_price_tiers(pricing.get("input_tiers")),
            cached_input_tiers=_parse_price_tiers(
                pricing.get("input_cache_read_tiers")
            ),
            cache_write_tiers=_parse_price_tiers(
                pricing.get("input_cache_write_tiers")
            ),
            output_tiers=_parse_price_tiers(pricing.get("output_tiers")),
        )
    return catalog


def _parse_price_tiers(value: Any) -> tuple[PriceTier, ...]:
    if not isinstance(value, list):
        return ()
    tiers: list[PriceTier] = []
    for item in value:
        if not isinstance(item, dict) or "cost" not in item:
            continue
        tiers.append(
            PriceTier(
                minimum=int(item.get("min", 0)),
                maximum=int(item["max"]) if item.get("max") is not None else None,
                cost_per_token=Decimal(str(item["cost"])),
            )
        )
    return tuple(tiers)


def parse_model_maps(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        source, separator, destination = value.partition("=")
        if not separator or not source.strip() or not destination.strip():
            raise ValueError("--model-map must use SOURCE=GATEWAY_ID")
        result[source.strip()] = destination.strip()
    return result


def resolve_price(
    provider: str,
    model: str,
    catalog: dict[str, ModelPrice],
    model_maps: dict[str, str],
) -> ModelPrice:
    mapped = model_maps.get(model) or model_maps.get(f"{provider}/{model}")
    candidates = [item for item in (mapped, model, f"{provider}/{model}") if item]
    for candidate in candidates:
        if candidate in catalog:
            return catalog[candidate]
    raise ValueError(
        f"No Vercel AI Gateway price for {provider}/{model}; "
        f"provide --model-map {model}=GATEWAY_ID"
    )


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def cost_for_usage(
    usage: dict[str, Any],
    *,
    provider: str,
    model: str,
    catalog: dict[str, ModelPrice],
    model_maps: dict[str, str],
) -> tuple[Decimal, ModelPrice]:
    """Price one model invocation from its usage object."""
    price = resolve_price(provider, model, catalog, model_maps)
    input_tokens = _usage_int(usage, "input_tokens")
    cached_tokens = _usage_int(usage, "cached_input_tokens")
    cache_write_tokens = _usage_int(usage, "cache_write_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
    cost = (
        Decimal(uncached_tokens) * price.input_rate(input_tokens)
        + Decimal(cached_tokens) * price.cached_input_rate(input_tokens)
        + Decimal(cache_write_tokens) * price.cache_write_rate(input_tokens)
        + Decimal(output_tokens) * price.output_rate(input_tokens)
    )
    return cost, price


def cost_from_token_usage(
    artifact_root: Path,
    catalog: dict[str, ModelPrice],
    model_maps: dict[str, str],
) -> tuple[Decimal, int, dict[str, ModelPrice]]:
    """Sum priced tokenUsage under an attempt artifact root.

    Returns ``(total_cost, priced_invocation_count, used_prices)``.
    """
    total_cost = Decimal(0)
    priced = 0
    used_prices: dict[str, ModelPrice] = {}
    usage_dir = artifact_root / "tokenUsage"
    usage_paths = sorted(usage_dir.glob("task-*.json")) if usage_dir.is_dir() else []
    for usage_path in usage_paths:
        usage_document = read_json(usage_path)
        attempts = usage_document.get("attempts", []) if isinstance(usage_document, dict) else []
        if not isinstance(attempts, list):
            raise ValueError(f"Invalid attempts in {usage_path}")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            invocations = attempt.get("invocations", [])
            if not isinstance(invocations, list):
                continue
            for invocation in invocations:
                if not isinstance(invocation, dict):
                    continue
                usage = invocation.get("usage")
                if not isinstance(usage, dict):
                    continue
                provider = str(invocation.get("provider", ""))
                model = str(invocation.get("model", ""))
                cost, price = cost_for_usage(
                    usage,
                    provider=provider,
                    model=model,
                    catalog=catalog,
                    model_maps=model_maps,
                )
                total_cost += cost
                priced += 1
                used_prices[price.gateway_id] = price
    return total_cost, priced, used_prices


def generate_report(
    run_name: str,
    *,
    results_root: Path,
    attempts_root: Path,
    catalog: dict[str, ModelPrice],
    model_maps: dict[str, str],
) -> dict[str, Any]:
    run_dir = results_root / run_name
    if not run_dir.is_dir():
        raise ValueError(f"Run results directory not found: {run_dir}")
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    result_paths = sorted(run_dir.glob("*.json"))
    result_paths = [path for path in result_paths if path.name != "summary.json"]
    results = [read_json(path) for path in result_paths]

    completed = 0
    errors = 0
    successful = 0
    trajectory_seconds: list[float] = []
    step_seconds: list[float] = []
    generation_seconds: list[float] = []
    browser_interaction_ms: list[float] = []
    steps_per_attempt: list[float] = []
    input_per_step: list[float] = []
    cached_per_step: list[float] = []
    cache_write_per_step: list[float] = []
    reasoning_per_step: list[float] = []
    non_reasoning_per_step: list[float] = []
    token_totals = {
        "total_tokens": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "non_reasoning_output_tokens": 0,
        "output_tokens": 0,
    }
    total_cost = Decimal(0)
    used_prices: dict[str, ModelPrice] = {}
    attempt_start_times: list[float] = []

    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Run result is not a JSON object")
        judgement = result.get("judgement")
        if isinstance(judgement, dict) and judgement.get("verdict") is True:
            successful += 1
        attempt_id = result.get("attempt_id")
        if not isinstance(attempt_id, str):
            raise ValueError("Run result is missing attempt_id")
        attempt_dir = attempts_root / attempt_id
        task_path = attempt_dir / "task.json"
        if task_path.is_file():
            attempt_start_times.append(task_path.stat().st_mtime)
        artifact_root = attempt_dir / "artifacts" / "browser-agent"
        trajectory_path = artifact_root / "steps.jsonl"
        if not trajectory_path.is_file():
            errors += 1
            continue
        try:
            trajectories = read_jsonl(trajectory_path)
        except ValueError:
            errors += 1
            continue
        error_file = artifact_root / "steps.error-tasks.json"
        recorded_errors = read_json(error_file) if error_file.is_file() else []
        crashed = bool(recorded_errors)
        if not trajectories:
            crashed = True
        if crashed:
            errors += 1
        if any(record.get("completed") is True for record in trajectories):
            completed += 1
        for trajectory in trajectories:
            duration_ms = trajectory.get("trajectoryDurationMs", trajectory.get("durationMs"))
            if isinstance(duration_ms, (int, float)):
                trajectory_seconds.append(duration_ms / 1000)
            runtime_metrics = trajectory.get("stepRuntimeMetrics", [])
            if isinstance(runtime_metrics, list):
                for metric in runtime_metrics:
                    if not isinstance(metric, dict):
                        continue
                    total_ms = metric.get("totalDurationMs")
                    generation_ms = metric.get("tokenGenerationMs")
                    browser_ms = metric.get("browserInteractionMs")
                    if isinstance(total_ms, (int, float)):
                        step_seconds.append(total_ms / 1000)
                    if isinstance(generation_ms, (int, float)):
                        generation_seconds.append(generation_ms / 1000)
                    if isinstance(browser_ms, (int, float)):
                        browser_interaction_ms.append(browser_ms)

        usage_dir = artifact_root / "tokenUsage"
        usage_paths = sorted(usage_dir.glob("task-*.json")) if usage_dir.is_dir() else []
        for usage_path in usage_paths:
            usage_document = read_json(usage_path)
            attempts = usage_document.get("attempts", []) if isinstance(usage_document, dict) else []
            if not isinstance(attempts, list):
                raise ValueError(f"Invalid attempts in {usage_path}")
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                invocations = attempt.get("invocations", [])
                if not isinstance(invocations, list):
                    continue
                executor_steps = 0
                for invocation in invocations:
                    if not isinstance(invocation, dict):
                        continue
                    usage = invocation.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    for key in token_totals:
                        token_totals[key] += _usage_int(usage, key)
                    provider = str(invocation.get("provider", ""))
                    model = str(invocation.get("model", ""))
                    cost, price = cost_for_usage(
                        usage,
                        provider=provider,
                        model=model,
                        catalog=catalog,
                        model_maps=model_maps,
                    )
                    used_prices[price.gateway_id] = price
                    total_cost += cost
                    input_tokens = _usage_int(usage, "input_tokens")
                    cached_tokens = _usage_int(usage, "cached_input_tokens")
                    cache_write_tokens = _usage_int(usage, "cache_write_tokens")
                    if invocation.get("kind") == "executor_step":
                        executor_steps += 1
                        input_per_step.append(input_tokens)
                        cached_per_step.append(cached_tokens)
                        cache_write_per_step.append(cache_write_tokens)
                        reasoning_per_step.append(_usage_int(usage, "reasoning_tokens"))
                        non_reasoning_per_step.append(
                            _usage_int(usage, "non_reasoning_output_tokens")
                        )
                steps_per_attempt.append(executor_steps)

    total = len(results)
    success_rate = round_int(successful / completed * 100) if completed else 0
    success_per_dollar = (
        Decimal(successful) / total_cost if total_cost > 0 else Decimal(0)
    )
    total_input = token_totals["input_tokens"]
    cached_inputs = token_totals["cached_input_tokens"]
    cache_write_inputs = token_totals["cache_write_tokens"]
    cache_hit_rate = round_int(cached_inputs / total_input * 100) if total_input else 0
    trajectory_distribution = distribution(trajectory_seconds)
    step_distribution = distribution(step_seconds)
    generation_distribution = distribution(generation_seconds)
    browser_distribution = distribution(browser_interaction_ms)
    browser_distribution["min"] = round_int(min(browser_interaction_ms)) if browser_interaction_ms else 0
    browser_distribution["max"] = round_int(max(browser_interaction_ms)) if browser_interaction_ms else 0
    browser_distribution = {
        "median": browser_distribution["median"],
        "mean": browser_distribution["mean"],
        "min": browser_distribution["min"],
        "max": browser_distribution["max"],
        "percentilesGraph": browser_distribution["percentilesGraph"],
    }

    return {
        "general": {
            "total": total,
            "completed": completed,
            "error": errors,
            "successful": successful,
            "successRate": f"{success_rate}%",
            "cost": f"${float(round_money(total_cost)):.2f}",
            "successPer$": round_money(success_per_dollar),
        },
        "prices": {
            model_id: used_prices[model_id].report_value()
            for model_id in sorted(used_prices)
        },
        "tokensAcrossAllTaskSteps": {
            "total": token_totals["total_tokens"],
            "totalInput": total_input,
            "cachedInputs": cached_inputs,
            "cacheWriteInputs": cache_write_inputs,
            "cacheHitRate": cache_hit_rate,
            "reasoning": token_totals["reasoning_tokens"],
            "nonReasoning": token_totals["non_reasoning_output_tokens"],
            "output": token_totals["output_tokens"],
        },
        "totalDurationSec": round_int(
            float(summary.get("duration_seconds", 0))
            if isinstance(summary, dict)
            else max(
                0.0,
                time.time()
                - min(attempt_start_times, default=run_dir.stat().st_mtime),
            )
        ),
        "averageTrajectoryDurationSec": (
            round_int(statistics.fmean(trajectory_seconds)) if trajectory_seconds else 0
        ),
        "timingSeconds": {
            "totalTrajectoryDuration": round_int(sum(trajectory_seconds)),
            "trajectoryDurationPerAttempt": trajectory_distribution,
            "stepDuration": step_distribution,
            "tokenGenerationPerStep": generation_distribution,
            "browserInteractionMsPerStep": browser_distribution,
        },
        "inputTokensPerStep": distribution(input_per_step, include_count=True),
        "cachedInputTokensPerStep": distribution(cached_per_step, include_count=True),
        "cacheWriteInputTokensPerStep": distribution(
            cache_write_per_step, include_count=True
        ),
        "reasoningTokensPerStep": distribution(reasoning_per_step, include_count=True),
        "nonReasoningOutputTokensPerStep": distribution(
            non_reasoning_per_step, include_count=True
        ),
        "stepsPerTaskAttempt": distribution(steps_per_attempt, include_count=True),
    }


def dump_report(report: dict[str, Any]) -> str:
    return yaml.dump(
        report,
        Dumper=ReportDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def omit_percentile_graphs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: omit_percentile_graphs(item)
            for key, item in value.items()
            if key != "percentilesGraph"
        }
    if isinstance(value, list):
        return [omit_percentile_graphs(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="benchmark run name")
    parser.add_argument(
        "--short",
        action="store_true",
        help="omit percentile graphs while retaining aggregate statistics",
    )
    parser.add_argument(
        "--model-map",
        action="append",
        default=[],
        metavar="SOURCE=GATEWAY_ID",
        help="map a local model alias to a Vercel AI Gateway model ID",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / ".runs" / "results"
    )
    parser.add_argument(
        "--attempts-dir", type=Path, default=REPO_ROOT / ".runs" / "attempts"
    )
    args = parser.parse_args()
    try:
        report = generate_report(
            args.name,
            results_root=args.results_dir,
            attempts_root=args.attempts_dir,
            catalog=fetch_gateway_catalog(),
            model_maps=parse_model_maps(args.model_map),
        )
        if args.short:
            report = omit_percentile_graphs(report)
        rendered = dump_report(report)
        sys.stdout.write(rendered)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
