from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_run.py"
SPEC = importlib.util.spec_from_file_location("report_run", SCRIPT)
assert SPEC and SPEC.loader
report_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report_run
SPEC.loader.exec_module(report_run)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    attempts = root / "attempts"
    run = results / "sample"
    write_json(run / "summary.json", {"duration_seconds": 20.4})
    write_json(
        run / "RBA-001.json",
        {
            "task_id": "RBA-001",
            "attempt_id": "sample-rba-001",
            "judgement": {"verdict": True},
        },
    )
    artifacts = attempts / "sample-rba-001" / "artifacts" / "browser-agent"
    trajectory = {
        "completed": True,
        "trajectoryDurationMs": 10000,
        "steps": [{"step": 1}, {"step": 2}],
        "stepRuntimeMetrics": [
            {
                "totalDurationMs": 2000,
                "tokenGenerationMs": 1500,
                "browserInteractionMs": 500,
            },
            {
                "totalDurationMs": 4000,
                "tokenGenerationMs": 3000,
                "browserInteractionMs": 1000,
            },
        ],
    }
    artifacts.mkdir(parents=True)
    (artifacts / "steps.jsonl").write_text(json.dumps(trajectory) + "\n")
    write_json(artifacts / "steps.error-tasks.json", [])
    invocations = [
        {
            "kind": "executor_step",
            "provider": "openai",
            "model": "test-model",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "reasoning_tokens": 10,
                "non_reasoning_output_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        },
        {
            "kind": "stage",
            "stage": "verifySuccess",
            "provider": "openai",
            "model": "test-model",
            "usage": {
                "input_tokens": 50,
                "cached_input_tokens": 0,
                "reasoning_tokens": 2,
                "non_reasoning_output_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 55,
            },
        },
    ]
    write_json(
        artifacts / "tokenUsage" / "task-001.json",
        {"attempts": [{"completed": True, "invocations": invocations}]},
    )
    return results, attempts


def price(cached: Decimal | None = Decimal("0.0000005")):
    return report_run.ModelPrice(
        gateway_id="openai/test-model",
        input_per_token=Decimal("0.000001"),
        cached_input_per_token=cached,
        cache_write_per_token=Decimal("0.00000125"),
        output_per_token=Decimal("0.000002"),
    )


class ReportRunTests(unittest.TestCase):
    def test_report_aggregates_metrics_cost_and_prices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, attempts = fixture(Path(temporary))
            report = report_run.generate_report(
                "sample",
                results_root=results,
                attempts_root=attempts,
                catalog={"openai/test-model": price()},
                model_maps={},
            )
        self.assertEqual(
            report["general"],
            {
                "total": 1,
                "completed": 1,
                "error": 0,
                "successful": 1,
                "successRate": "100%",
                "cost": "$0.00",
                "successPer$": 5555.56,
            },
        )
        self.assertEqual(
            report["tokensAcrossAllTaskSteps"],
            {
                "total": 175,
                "totalInput": 150,
                "cachedInputs": 40,
                "cacheWriteInputs": 0,
                "cacheHitRate": 27,
                "reasoning": 12,
                "nonReasoning": 13,
                "output": 25,
            },
        )
        self.assertEqual(report["totalDurationSec"], 20)
        self.assertEqual(report["averageTrajectoryDurationSec"], 10)
        self.assertEqual(report["inputTokensPerStep"]["count"], 1)
        self.assertEqual(
            report["prices"]["openai/test-model"],
            {
                "inputPerM": 1.0,
                "cachedInputPerM": 0.5,
                "cacheWriteInputPerM": 1.25,
                "outputPerM": 2.0,
            },
        )

    def test_report_charges_cache_writes_separately_from_uncached_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, attempts = fixture(Path(temporary))
            usage_path = (
                attempts
                / "sample-rba-001"
                / "artifacts"
                / "browser-agent"
                / "tokenUsage"
                / "task-001.json"
            )
            usage_document = json.loads(usage_path.read_text(encoding="utf-8"))
            invocations = usage_document["attempts"][0]["invocations"]
            invocations[0]["usage"].update(
                {
                    "input_tokens": 1_000_000,
                    "cached_input_tokens": 200_000,
                    "cache_write_tokens": 300_000,
                    "output_tokens": 100_000,
                    "total_tokens": 1_100_000,
                }
            )
            invocations[1]["usage"].update(
                {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            )
            write_json(usage_path, usage_document)

            report = report_run.generate_report(
                "sample",
                results_root=results,
                attempts_root=attempts,
                catalog={"openai/test-model": price()},
                model_maps={},
            )

        # 500k regular input + 200k cache reads + 300k cache writes + 100k output.
        self.assertEqual(report["general"]["cost"], "$1.18")
        self.assertEqual(
            report["tokensAcrossAllTaskSteps"]["cacheWriteInputs"], 300_000
        )
        self.assertEqual(report["cacheWriteInputTokensPerStep"]["count"], 1)

    def test_gateway_catalog_reads_cache_write_base_and_tiers(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openai/cache-model",
                    "pricing": {
                        "input": "0.000001",
                        "input_cache_read": "0.0000001",
                        "input_cache_write": "0.00000125",
                        "input_cache_write_tiers": [
                            {
                                "min": 100_000,
                                "max": None,
                                "cost": "0.00000375",
                            }
                        ],
                        "output": "0.000002",
                    },
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch.object(
            report_run.urllib.request, "urlopen", return_value=FakeResponse()
        ):
            model = report_run.fetch_gateway_catalog()["openai/cache-model"]

        self.assertEqual(model.cache_write_per_token, Decimal("0.00000125"))
        self.assertEqual(model.cache_write_rate(99_999), Decimal("0.00000125"))
        self.assertEqual(model.cache_write_rate(100_000), Decimal("0.00000375"))

    def test_report_counts_missing_trajectory_as_execution_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = fixture(root)
            write_json(
                results / "sample" / "RBA-002.json",
                {
                    "task_id": "RBA-002",
                    "attempt_id": "sample-rba-002",
                    "judgement": None,
                },
            )
            report = report_run.generate_report(
                "sample",
                results_root=results,
                attempts_root=attempts,
                catalog={"openai/test-model": price()},
                model_maps={},
            )
        self.assertEqual(report["general"]["total"], 2)
        self.assertEqual(report["general"]["completed"], 1)
        self.assertEqual(report["general"]["error"], 1)

    def test_report_works_without_summary_for_an_intermediate_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, attempts = fixture(Path(temporary))
            summary = results / "sample" / "summary.json"
            summary.unlink()
            started_at = time.time() - 12.6
            with patch.object(report_run.time, "time", return_value=time.time()):
                summary.parent.touch()
                # Set the directory timestamp after touch so elapsed duration is stable.
                import os

                os.utime(summary.parent, (started_at, started_at))
                report = report_run.generate_report(
                    "sample",
                    results_root=results,
                    attempts_root=attempts,
                    catalog={"openai/test-model": price()},
                    model_maps={},
                )
        self.assertEqual(report["general"]["total"], 1)
        self.assertEqual(report["general"]["completed"], 1)
        self.assertEqual(report["totalDurationSec"], 13)

    def test_cached_tokens_fall_back_to_input_price(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, attempts = fixture(Path(temporary))
            report = report_run.generate_report(
                "sample",
                results_root=results,
                attempts_root=attempts,
                catalog={"openai/test-model": price(None)},
                model_maps={},
            )
        self.assertIsNone(
            report["prices"]["openai/test-model"]["cachedInputPerM"]
        )

    def test_percentile_graph_and_yaml_money_format(self) -> None:
        graph = report_run.percentile_graph([0, 10])
        self.assertTrue(graph.splitlines()[0].startswith("  p0 |"))
        self.assertTrue(graph.splitlines()[-1].endswith("#" * 30 + " 10"))
        rendered = report_run.dump_report(
            {"cost": report_run.round_money(Decimal("1.2")), "graph": graph}
        )
        self.assertIn("cost: 1.20", rendered)
        self.assertIn("graph: |2-\n", rendered)
        self.assertEqual(yaml.safe_load(rendered)["cost"], 1.2)

    def test_model_alias_requires_or_uses_mapping(self) -> None:
        catalog = {"openai/real-model": price()}
        with self.assertRaisesRegex(ValueError, "--model-map"):
            report_run.resolve_price("openai", "alias", catalog, {})
        resolved = report_run.resolve_price(
            "openai", "alias", catalog, {"alias": "openai/real-model"}
        )
        self.assertEqual(resolved.gateway_id, "openai/test-model")

    def test_tiered_model_uses_invocation_input_size(self) -> None:
        model = report_run.ModelPrice(
            gateway_id="openai/tiered",
            input_per_token=Decimal("0.000001"),
            cached_input_per_token=Decimal("0.0000001"),
            cache_write_per_token=Decimal("0.00000125"),
            output_per_token=Decimal("0.000002"),
            input_tiers=(
                report_run.PriceTier(0, 100, Decimal("0.000001")),
                report_run.PriceTier(100, None, Decimal("0.000003")),
            ),
            cached_input_tiers=(
                report_run.PriceTier(100, None, Decimal("0.0000003")),
            ),
            cache_write_tiers=(
                report_run.PriceTier(100, None, Decimal("0.00000375")),
            ),
            output_tiers=(
                report_run.PriceTier(100, None, Decimal("0.000006")),
            ),
        )
        self.assertEqual(model.input_rate(99), Decimal("0.000001"))
        self.assertEqual(model.input_rate(100), Decimal("0.000003"))
        self.assertEqual(model.cached_input_rate(100), Decimal("0.0000003"))
        self.assertEqual(model.cache_write_rate(100), Decimal("0.00000375"))
        self.assertEqual(model.output_rate(100), Decimal("0.000006"))

    def test_short_report_omits_all_percentile_graphs(self) -> None:
        report = {
            "timing": {
                "mean": 5,
                "percentilesGraph": "graph",
                "nested": {"percentilesGraph": "another", "median": 4},
            },
            "values": [{"percentilesGraph": "third", "count": 2}],
        }
        shortened = report_run.omit_percentile_graphs(report)
        self.assertEqual(
            shortened,
            {
                "timing": {"mean": 5, "nested": {"median": 4}},
                "values": [{"count": 2}],
            },
        )


if __name__ == "__main__":
    unittest.main()
