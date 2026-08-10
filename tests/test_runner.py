from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from rbbench.catalog import Catalog
from rbbench.errors import ExecutorError
from rbbench.executors import Executor
from rbbench.judges import Judge
from rbbench.io import write_json
from rbbench.runner import BenchmarkRunner
from rbbench.schema import (
    AttemptDescriptor,
    ExecutionResult,
    JudgementResult,
    RunStatus,
    TaskSpec,
)


class PassingExecutor(Executor):
    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        return ExecutionResult(
            final_result="42",
            observation={
                "result": {"primary": "42", "details": {"unit": "widgets"}},
                "safety": {"forbidden_action_performed": False},
            },
        )


class SessionCaptureExecutor(PassingExecutor):
    def __init__(self) -> None:
        self.sessions: list[dict[str, object]] = []

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        self.sessions.append(dict(attempt.session))
        return await super().execute(task, attempt)


class CountingExecutor(PassingExecutor):
    def __init__(self) -> None:
        self.calls = 0
        self.prepare_calls = 0

    async def prepare_run(self) -> None:
        self.prepare_calls += 1

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        self.calls += 1
        return await super().execute(task, attempt)


class PreparingExecutor(CountingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.prepared = False

    async def prepare_run(self) -> None:
        await super().prepare_run()
        await asyncio.sleep(0)
        self.prepared = True

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        if not self.prepared:
            raise AssertionError("attempt started before run preflight")
        await asyncio.sleep(0)
        return await super().execute(task, attempt)


class FailingPreparationExecutor(CountingExecutor):
    async def prepare_run(self) -> None:
        self.prepare_calls += 1
        raise ExecutorError("preflight failed")


class PassingJudge(Judge):
    async def evaluate(self, task, attempt, execution, trusted_observation):
        return JudgementResult(reasoning="trusted evidence matches", verdict=True)


class MissingReferenceJudge(Judge):
    async def evaluate(self, task, attempt, execution, trusted_observation):
        from rbbench.errors import InvalidEnvironmentError

        raise InvalidEnvironmentError("Missing reference ground truth")


class RejectingJudge(Judge):
    async def evaluate(self, task, attempt, execution, trusted_observation):
        return JudgementResult(
            reasoning="required fact was missing",
            verdict=False,
            failure_reason="Missing requested fact",
        )


def public_task(task_id: str = "RBA-999") -> TaskSpec:
    return TaskSpec.from_dict(
        {
            "task_id": task_id,
            "title": "Runner fixture",
            "confirmed_task": "Return the fixed test value.",
            "category": "test",
            "environment": {
                "adapter": "public_web",
                "kind": "production_public_web",
                "start_url": "https://example.com/",
            },
            "fixture": {},
            "oracle": {
                "type": "reference_result",
                "reference_key": task_id,
                "assertions": [
                    {
                        "kind": "equals",
                        "path": "result.primary",
                        "expected_path": "result.primary",
                    },
                    {
                        "kind": "equals",
                        "path": "result.details",
                        "expected_path": "result.details",
                    },
                    {
                        "kind": "falsey",
                        "path": "safety.forbidden_action_performed",
                    },
                ],
            },
            "cleanup": {"strategy": "close_fresh_context"},
            "safety": {},
            "sources": ["https://example.com/"],
        }
    )


class RunnerTests(unittest.TestCase):
    def test_prepares_once_before_parallel_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = (public_task("RBA-997"), public_task("RBA-998"))
            executor = PreparingExecutor()
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", tasks, root / "catalog.json"),
                executor=executor,
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )

            asyncio.run(
                runner.run(tasks, parallel=2, run_id="parallel-preflight")
            )

            self.assertEqual(executor.prepare_calls, 1)
            self.assertEqual(executor.calls, 2)

    def test_failed_preflight_starts_no_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            executor = FailingPreparationExecutor()
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=executor,
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )

            with self.assertRaisesRegex(ExecutorError, "preflight failed"):
                asyncio.run(runner.run([task], run_id="failed-preflight"))

            self.assertEqual(executor.prepare_calls, 1)
            self.assertEqual(executor.calls, 0)

    def test_successful_llm_judged_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            catalog = Catalog("test", "1", (task,), root / "catalog.json")
            reference_dir = root / "references"
            write_json(
                reference_dir / "RBA-999.json",
                {"result": {"primary": "42", "details": {"unit": "widgets"}}},
            )
            runner = BenchmarkRunner(
                catalog=catalog,
                executor=PassingExecutor(),
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            summary = asyncio.run(
                runner.run([task], parallel=1, run_id="test-run")
            )
            self.assertEqual(summary["status_counts"], {"success": 1})
            self.assertEqual(summary["mean_score"], 1.0)
            result = summary["results"][0]
            self.assertEqual(result["status"], RunStatus.SUCCESS.value)
            self.assertTrue(result["judgement"]["verdict"])
            self.assertTrue((root / "results/test-run/summary.json").exists())
            self.assertTrue(
                (root / "attempts/test-run-rba-999/trusted-observation.json").exists()
            )

    def test_resume_reuses_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            executor = CountingExecutor()
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=executor,
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            first = asyncio.run(runner.run([task], run_id="test-run"))
            resumed = asyncio.run(
                runner.run([task], run_id="test-run", resume=True)
            )
            self.assertEqual(executor.calls, 1)
            self.assertEqual(executor.prepare_calls, 1)
            self.assertEqual(resumed["results"], first["results"])

    def test_resume_preserves_incomplete_attempt_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            old_trace = root / "attempts/test-run-rba-999/artifacts/steps.jsonl"
            old_trace.parent.mkdir(parents=True)
            old_trace.write_text("existing trace", encoding="utf-8")
            (root / "results/test-run").mkdir(parents=True)
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=PassingExecutor(),
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            resumed = asyncio.run(
                runner.run([task], run_id="test-run", resume=True)
            )
            self.assertEqual(resumed["task_count"], 1)
            self.assertEqual(old_trace.read_text(encoding="utf-8"), "existing trace")
            self.assertTrue(
                (root / "attempts/test-run-rba-999-resume-2/result.json").exists()
            )

    def test_missing_reference_is_not_an_agent_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            catalog = Catalog("test", "1", (task,), root / "catalog.json")
            runner = BenchmarkRunner(
                catalog=catalog,
                executor=PassingExecutor(),
                judge=MissingReferenceJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            result = asyncio.run(runner.run_one(task, attempt_id="missing-ref"))
            self.assertEqual(result.status, RunStatus.INVALID_ENVIRONMENT)
            self.assertIn("Missing reference ground truth", result.error or "")

    def test_false_judge_verdict_is_agent_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = public_task()
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=PassingExecutor(),
                judge=RejectingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            result = asyncio.run(runner.run_one(task, attempt_id="rejected"))
            self.assertEqual(result.status, RunStatus.AGENT_FAILURE)
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.error, "Missing requested fact")

    def test_browser_profile_is_seeded_into_attempt_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "chrome-profile"
            profile.mkdir()
            task = public_task()
            executor = SessionCaptureExecutor()
            runner = BenchmarkRunner(
                catalog=Catalog("test", "1", (task,), root / "catalog.json"),
                executor=executor,
                judge=PassingJudge(),
                runtime_dir=root / "attempts",
                results_dir=root / "results",
                browser_profile=profile,
            )
            result = asyncio.run(runner.run_one(task, attempt_id="with-profile"))
            self.assertEqual(result.status, RunStatus.SUCCESS)
            self.assertEqual(len(executor.sessions), 1)
            self.assertEqual(
                executor.sessions[0]["user_data_dir"],
                str(profile.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
