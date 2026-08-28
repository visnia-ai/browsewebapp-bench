from __future__ import annotations

import contextlib
import asyncio
import io
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from rbbench.catalog import load_catalog
from rbbench.cli import build_parser, main
from rbbench.errors import ExecutorError
from rbbench.executors import (
    BrowserAgentExecutor,
    CommandExecutor,
    _browser_agent_instruction,
    _stage_inputs,
)
from rbbench.schema import AttemptDescriptor


class InputStagingTests(unittest.TestCase):
    def test_staged_inputs_use_workspace_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "certificate.pdf"
            source.parent.mkdir()
            source.write_bytes(b"%PDF-1.4\nfixture")
            artifact_dir = root / "artifacts"

            staged = _stage_inputs({"input_artifact": str(source)}, artifact_dir)

            self.assertEqual(
                staged, {"input_artifact": "./inputs/certificate.pdf"}
            )
            self.assertEqual(
                (artifact_dir / "inputs" / "certificate.pdf").read_bytes(),
                source.read_bytes(),
            )

    def test_instruction_uses_attempt_scope(self) -> None:
        task = load_catalog().by_id("RBA-015")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = AttemptDescriptor(
                "attempt-123",
                task.task_id,
                task.environment.start_url,
                root,
                root / "artifacts",
            )
            prompt = _browser_agent_instruction(task, attempt, {})
        self.assertIn("Attempt ID: attempt-123", prompt)
        self.assertIn(task.confirmed_task, prompt)


class BrowserAgentExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_process_starts_are_spaced_to_avoid_port_races(self) -> None:
        starts: list[float] = []

        class Agent:
            def run(self, task):
                starts.append(asyncio.get_running_loop().time())
                return task

        executor = BrowserAgentExecutor()
        executor._SDK_LAUNCH_INTERVAL_SECONDS = 0.02

        results = await asyncio.gather(
            executor._start_sdk_run(Agent(), "one"),
            executor._start_sdk_run(Agent(), "two"),
            executor._start_sdk_run(Agent(), "three"),
        )

        self.assertEqual(results, ["one", "two", "three"])
        self.assertGreaterEqual(starts[1] - starts[0], 0.015)
        self.assertGreaterEqual(starts[2] - starts[1], 0.015)

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    async def test_command_timeout_terminates_descendant_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "child-terminated"
            child = root / "child.py"
            child.write_text(
                "import os\n"
                "import signal\n"
                "import sys\n"
                "import time\n"
                "from pathlib import Path\n"
                "def stop(_signum, _frame):\n"
                "    Path(os.environ['CHILD_TERMINATED']).write_text('yes')\n"
                "    raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "Path(os.environ['CHILD_READY']).write_text('yes')\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            adapter = root / "adapter.py"
            adapter.write_text(
                "import os\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "subprocess.Popen([sys.executable, os.environ['CHILD_SCRIPT']])\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            task = load_catalog().by_id("RBA-015")
            attempt = AttemptDescriptor(
                "command-timeout",
                task.task_id,
                task.environment.start_url,
                root / "attempt",
                root / "attempt" / "artifacts",
            )
            attempt.attempt_dir.mkdir(parents=True)
            executor = CommandExecutor(
                f"{sys.executable} {adapter}", timeout_seconds=1
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "CHILD_SCRIPT": str(child),
                        "CHILD_READY": str(root / "child-ready"),
                        "CHILD_TERMINATED": str(marker),
                    },
                ),
                self.assertRaisesRegex(ExecutorError, "timed out after 1s"),
            ):
                await executor.execute(task, attempt)

            self.assertTrue(marker.is_file())

    async def test_codex_preflight_checks_sdk_once_without_login(self) -> None:
        calls: list[float] = []

        async def check_codex_login(*, timeout_seconds):
            calls.append(timeout_seconds)
            return True

        module = types.ModuleType("browser_agent")
        module.check_codex_login = check_codex_login
        with patch.dict(sys.modules, {"browser_agent": module}):
            await BrowserAgentExecutor(
                provider="codex",
                model="gpt-5.6-luna",
                timeout_seconds=123,
            ).prepare_run()

        self.assertEqual(calls, [123])

    async def test_codex_preflight_rejects_logged_out_before_execution(self) -> None:
        async def check_codex_login(*, timeout_seconds):
            return False

        module = types.ModuleType("browser_agent")
        module.check_codex_login = check_codex_login
        with (
            patch.dict(sys.modules, {"browser_agent": module}),
            self.assertRaisesRegex(
                ExecutorError,
                r"Codex login is required\. Run 'rbbench codex-login' first, then retry\.",
            ),
        ):
            await BrowserAgentExecutor(provider="codex").prepare_run()

    async def test_non_codex_preflight_is_a_noop(self) -> None:
        await BrowserAgentExecutor().prepare_run()

    async def test_cli_accepts_codex_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--name",
                "codex-login",
                "--provider",
                "codex",
                "--model",
                "gpt-5.6-luna",
            ]
        )

        self.assertEqual(args.provider, "codex")

    def test_cli_codex_login_runs_interactive_sdk_and_relays_url(self) -> None:
        calls = 0

        async def ensure_codex_login(*, on_log):
            nonlocal calls
            calls += 1
            on_log(types.SimpleNamespace(message="https://auth.example.test/login"))

        module = types.ModuleType("browser_agent")
        module.ensure_codex_login = ensure_codex_login
        stderr = io.StringIO()
        with (
            patch.dict(sys.modules, {"browser_agent": module}),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["codex-login"])

        self.assertEqual(result, 0)
        self.assertEqual(calls, 1)
        self.assertIn("https://auth.example.test/login", stderr.getvalue())

    async def test_sdk_receives_glm_baseline_options(self) -> None:
        captured: dict[str, object] = {}

        @dataclass
        class Credential:
            username: str
            password: str
            domain: str

        @dataclass
        class Task:
            task: str
            url: str | None = None
            credentials: tuple[Credential, ...] = ()

        class Agent:
            def __init__(self, **kwargs):
                captured["options"] = kwargs

            def run(self, task):
                captured["task"] = task
                captured["options"]["on_log"](
                    types.SimpleNamespace(
                        message="Codex ChatGPT sign-in is required."
                    )
                )
                result = types.SimpleNamespace(
                    run_id="sdk-run",
                    status="completed",
                    tasks=(
                        types.SimpleNamespace(
                            status="completed",
                            runs=(
                                types.SimpleNamespace(
                                    completed=True,
                                    data={"answer": "ok"},
                                ),
                            ),
                            errors=(),
                        ),
                    ),
                )
                import asyncio

                future = asyncio.get_running_loop().create_future()
                future.set_result(result)
                return types.SimpleNamespace(result=future, cancel=lambda: None)

        module = types.ModuleType("browser_agent")
        module.BrowserAgent = Agent
        module.BrowserAgentCredential = Credential
        module.BrowserAgentTask = Task

        task = load_catalog().by_id("RBA-015")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"browser_agent": module}
        ):
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            attempt = AttemptDescriptor(
                "attempt-sdk",
                task.task_id,
                task.environment.start_url,
                root,
                artifacts,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = await BrowserAgentExecutor(
                    endpoint_url="http://model.test:8001/v1"
                ).execute(task, attempt)

        options = captured["options"]
        self.assertEqual(options["provider"], "vllm")
        self.assertEqual(options["model"], "nvidia/GLM-5.2-NVFP4")
        self.assertEqual(options["reasoning_effort"], "high")
        self.assertEqual(options["max_model_len"], 48_000)
        self.assertEqual(options["reserve_output_tokens"], 4_000)
        self.assertTrue(callable(options["on_log"]))
        self.assertIn("Codex ChatGPT sign-in is required.", stderr.getvalue())
        self.assertFalse(options["headless"])
        self.assertEqual(options["max_steps"], 50)
        self.assertEqual(result.final_result, '{"answer": "ok"}')


if __name__ == "__main__":
    unittest.main()
