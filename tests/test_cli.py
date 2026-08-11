from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rbbench.catalog import Catalog
from rbbench.cli import (
    _browser_profile,
    _confirm_benchmark_overwrite,
    _existing_benchmark_paths,
    _judge,
    _selected,
    _validate_benchmark_name,
    build_parser,
)
from rbbench.io import write_json
from rbbench.judges import NativeLLMAdapter, NativeLLMJudge
from rbbench.runner import BenchmarkRunner

from test_runner import PassingExecutor, public_task


class CliSelectionTests(unittest.TestCase):
    def test_run_requires_name(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["run"])

    def test_run_accepts_name(self) -> None:
        args = build_parser().parse_args(["run", "--name", "glm-baseline.1"])
        self.assertEqual(args.name, "glm-baseline.1")

    def test_run_accepts_browser_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "chrome-profile"
            profile.mkdir()
            args = build_parser().parse_args(
                [
                    "run",
                    "--name",
                    "profile-run",
                    "--browser-profile",
                    str(profile),
                ]
            )
            self.assertEqual(_browser_profile(args.browser_profile), profile.resolve())

    def test_browser_profile_requires_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-profile"
            with self.assertRaisesRegex(ValueError, "--browser-profile"):
                _browser_profile(missing)

    def test_benchmark_name_rejects_paths(self) -> None:
        for name in ("../run", "/tmp/run", "run/name", "run name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_benchmark_name(name)

    def test_overwrite_removes_results_and_exact_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            attempts = root / "attempts"
            result_path = results / "baseline"
            attempt_path = attempts / "baseline-rba-001"
            similarly_named = attempts / "baseline-extra-rba-001"
            for path in (result_path, attempt_path, similarly_named):
                path.mkdir(parents=True)
                (path / "data").write_text("test", encoding="utf-8")
            paths = _existing_benchmark_paths(
                "baseline",
                results_dir=results,
                runtime_dir=attempts,
                task_ids=["RBA-001"],
            )
            with patch("builtins.input", return_value="overwrite"):
                action = _confirm_benchmark_overwrite("baseline", paths)
            self.assertEqual(action, "overwrite")
            self.assertFalse(result_path.exists())
            self.assertFalse(attempt_path.exists())
            self.assertTrue(similarly_named.exists())

    def test_resume_preserves_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results" / "baseline"
            path.mkdir(parents=True)
            with patch("builtins.input", return_value="resume"):
                action = _confirm_benchmark_overwrite("baseline", [path])
            self.assertEqual(action, "resume")
            self.assertTrue(path.exists())

    def test_existing_benchmark_rejects_yes_or_no(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            path.mkdir()
            for answer in ("yes", "no"):
                with self.subTest(answer=answer):
                    with patch("builtins.input", return_value=answer):
                        with self.assertRaisesRegex(
                            ValueError, "choose 'overwrite' or 'resume'"
                        ):
                            _confirm_benchmark_overwrite("baseline", [path])

    def test_adapter_filter_selects_public_web_tasks(self) -> None:
        _, tasks = _selected(
            argparse.Namespace(
                catalog=None,
                task=None,
                adapter="public_web",
                category=None,
            )
        )
        self.assertEqual(len(tasks), 80)
        self.assertTrue(
            all(task.environment.adapter == "public_web" for task in tasks)
        )

    def test_category_filter_composes_with_adapter(self) -> None:
        _, tasks = _selected(
            argparse.Namespace(
                catalog=None,
                task=None,
                adapter="public_web",
                category="registry/equipment_authorization",
            )
        )
        self.assertTrue(tasks)
        self.assertTrue(
            all(
                task.environment.adapter == "public_web"
                and task.category == "registry/equipment_authorization"
                for task in tasks
            )
        )

class DefaultJudgeSmokeTests(unittest.TestCase):
    def test_bare_run_uses_inline_openai_luna_judge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            reference_dir = root / "references"
            write_json(
                reference_dir / "RBA-999.json",
                {"result": {"primary": "42", "details": {"unit": "widgets"}}},
            )
            args = build_parser().parse_args(["run", "--name", "judge-smoke"])
            args.reference_dir = reference_dir
            with patch.dict(
                "os.environ", {"OPENAI_API_KEY": "test-openai-key"}
            ):
                judge = _judge(args)

            self.assertIsInstance(judge, NativeLLMJudge)
            self.assertEqual(judge.provider, "openai")
            self.assertEqual(judge.model, "gpt-5.6-luna")
            self.assertEqual(judge.base_url, "https://api.openai.com/v1")
            self.assertEqual(judge.reasoning_effort, "high")
            self.assertTrue(judge.text_only)
            self.assertEqual(judge.request_extra_body, {})

            requests: list[dict[str, object]] = []

            def fake_request(adapter, payload, headers):
                requests.append(
                    {
                        "url": adapter.url,
                        "payload": payload,
                        "headers": headers,
                    }
                )
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "reasoning": "trusted evidence matches",
                                        "verdict": True,
                                        "failure_reason": "",
                                        "impossible_task": False,
                                        "reached_captcha": False,
                                    }
                                )
                            }
                        }
                    ]
                }

            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=PassingExecutor(),
                judge=judge,
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            with patch.object(NativeLLMAdapter, "_request", fake_request):
                summary = asyncio.run(
                    runner.run([task], parallel=1, run_id="default-judge-smoke")
                )

            self.assertEqual(summary["status_counts"], {"success": 1})
            self.assertEqual(len(requests), 1)
            self.assertEqual(
                requests[0]["url"],
                "https://api.openai.com/v1/chat/completions",
            )
            payload = requests[0]["payload"]
            self.assertEqual(payload["model"], "gpt-5.6-luna")
            self.assertNotIn("provider", payload)
            self.assertEqual(payload["max_completion_tokens"], 4000)
            self.assertEqual(payload["reasoning_effort"], "high")
            judgement = summary["results"][0]["judgement"]
            self.assertEqual(judgement["provider"], "openai")
            self.assertEqual(judgement["model"], "gpt-5.6-luna")
            self.assertTrue(judgement["verdict"])


if __name__ == "__main__":
    unittest.main()
