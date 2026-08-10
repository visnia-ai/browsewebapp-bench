from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summary.py"
SPEC = importlib.util.spec_from_file_location("summary_script", SCRIPT)
assert SPEC and SPEC.loader
summary_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_script
SPEC.loader.exec_module(summary_script)

report_run = summary_script.report_run


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def price(cached: Decimal | None = Decimal("0.0000005")):
    return report_run.ModelPrice(
        gateway_id="openai/test-model",
        input_per_token=Decimal("0.000001"),
        cached_input_per_token=cached,
        cache_write_per_token=Decimal("0.00000125"),
        output_per_token=Decimal("0.000002"),
    )


class SummaryScriptTests(unittest.TestCase):
    def make_run(self, root: Path, name: str, *, direct: bool = True) -> tuple[Path, Path]:
        results = root / "results"
        attempts = root / "attempts"
        attempt_id = f"{name}-rba-001"
        metrics = (
            {
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "steps": 6,
            }
            if direct
            else {"duration_seconds": 10}
        )
        write_json(
            results / name / "RBA-001.json",
            {
                "attempt_id": attempt_id,
                "score": 1,
                "judgement": {"verdict": True},
                "duration_seconds": 12.6,
                "execution": {"metrics": metrics},
            },
        )
        write_json(results / name / "summary.json", {"duration_seconds": 12.6})
        write_json(attempts / attempt_id / "task.json", {"task_id": "RBA-001"})
        return results, attempts

    def test_direct_metrics_support_browser_use_and_bcode_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, attempts = self.make_run(Path(temporary), "direct")
            row = summary_script.summarize_run(
                "direct", results_root=results, attempts_root=attempts
            )
        self.assertEqual(
            row,
            [
                "direct",
                "100.00%",
                "1,000",
                "400",
                "200",
                "50",
                "6",
                "13s",
                "N/A",
                "N/A",
            ],
        )

    def test_browser_agent_falls_back_to_token_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = self.make_run(root, "browser-agent", direct=False)
            usage = {
                "totals": {
                    "input_tokens": 3000,
                    "cached_input_tokens": 1200,
                    "output_tokens": 500,
                    "reasoning_tokens": 125,
                },
                "attempts": [
                    {
                        "invocations": [
                            {"kind": "executor_step"},
                            {"kind": "executor_step"},
                            {"kind": "stage"},
                        ]
                    }
                ],
            }
            write_json(
                attempts
                / "browser-agent-rba-001"
                / "artifacts"
                / "browser-agent"
                / "tokenUsage"
                / "task-001.json",
                usage,
            )
            row = summary_script.summarize_run(
                "browser-agent", results_root=results, attempts_root=attempts
            )
        self.assertEqual(row[2:7], ["3,000", "1,200", "500", "125", "2"])
        self.assertEqual(row[-2:], ["N/A", "N/A"])

    def test_token_usage_pricing_sets_cost_and_success_per_dollar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = self.make_run(root, "priced", direct=False)
            usage = {
                "attempts": [
                    {
                        "invocations": [
                            {
                                "kind": "executor_step",
                                "provider": "openai",
                                "model": "test-model",
                                "usage": {
                                    "input_tokens": 1_000_000,
                                    "cached_input_tokens": 200_000,
                                    "cache_write_tokens": 0,
                                    "output_tokens": 100_000,
                                },
                            }
                        ]
                    }
                ],
            }
            write_json(
                attempts
                / "priced-rba-001"
                / "artifacts"
                / "browser-agent"
                / "tokenUsage"
                / "task-001.json",
                usage,
            )
            row = summary_script.summarize_run(
                "priced",
                results_root=results,
                attempts_root=attempts,
                catalog={"openai/test-model": price()},
            )
        # 800k uncached * 1e-6 + 200k cached * 5e-7 + 100k out * 2e-6 = 0.8 + 0.1 + 0.2
        self.assertEqual(row[-2:], ["$1.10", "0.91"])

    def test_metrics_cost_fallback_when_token_usage_unpriced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = self.make_run(root, "metrics-cost")
            result_path = results / "metrics-cost" / "RBA-001.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["execution"]["metrics"]["cost"] = 0.25
            write_json(result_path, result)
            row = summary_script.summarize_run(
                "metrics-cost", results_root=results, attempts_root=attempts
            )
        self.assertEqual(row[-2:], ["$0.25", "4.00"])

    def test_zero_cost_reports_na_success_per_dollar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = self.make_run(root, "zero-cost")
            result_path = results / "zero-cost" / "RBA-001.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["execution"]["metrics"]["cost"] = 0.0
            write_json(result_path, result)
            row = summary_script.summarize_run(
                "zero-cost", results_root=results, attempts_root=attempts
            )
        self.assertEqual(row[-2:], ["$0.00", "N/A"])

    def test_duration_is_na_when_a_task_duration_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            attempts = root / "attempts"
            write_json(
                results / "live" / "RBA-001.json",
                {"attempt_id": "live-rba-001", "judgement": {"verdict": False}},
            )
            row = summary_script.summarize_run(
                "live", results_root=results, attempts_root=attempts, now=112.4
            )
        self.assertEqual(
            row,
            [
                "live",
                "0.00%",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            ],
        )

    def test_duration_sums_task_durations_not_benchmark_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, attempts = self.make_run(root, "summed")
            write_json(
                results / "summed" / "RBA-002.json",
                {
                    "attempt_id": "summed-rba-002",
                    "score": 0,
                    "execution": {"metrics": {"duration_seconds": 7.6}},
                },
            )
            row = summary_script.summarize_run(
                "summed", results_root=results, attempts_root=attempts
            )
        self.assertEqual(row[7], "20s")

    def test_multiple_names_render_ordered_markdown_rows(self) -> None:
        rows = [
            ["first", "50.00%", "1", "0", "2", "1", "3", "4s", "$1.00", "1.00"],
            ["second", "75.00%", "10", "5", "4", "2", "6", "8s", "$2.00", "0.50"],
        ]
        rendered = summary_script.render_table(rows)
        lines = rendered.splitlines()
        self.assertIn("| Benchmark", lines[0])
        self.assertIn("Cost", lines[0])
        self.assertIn("Successful tasks / $", lines[0])
        self.assertIn("first", lines[2])
        self.assertIn("second", lines[3])
        self.assertLess(rendered.index("first"), rendered.index("second"))

    def test_parse_comma_separated_names(self) -> None:
        self.assertEqual(
            summary_script.parse_names(" first, second ,third"),
            ["first", "second", "third"],
        )
        with self.assertRaises(ValueError):
            summary_script.parse_names("first,,third")


if __name__ == "__main__":
    unittest.main()
