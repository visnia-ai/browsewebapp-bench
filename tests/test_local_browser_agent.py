from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rbbench.catalog import load_catalog
from rbbench.cli import _executor, build_parser
from rbbench.errors import ExecutorError
from rbbench.local_browser_agent import (
    LocalBrowserAgentExecutor,
    resolve_local_cli_command,
)
from rbbench.schema import AttemptDescriptor


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _base_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "stage_llms": {
                    "runAgent": {
                        "provider": "openrouter",
                        "model": "z-ai/glm-5.2",
                    }
                },
                "feature_flags": {"semantic_projection_history": "current"},
                "headless": True,
                "max_steps": 37,
                "task_run_retry_count": 2,
                "concurrency": 8,
                "tasks": [{"task": "placeholder"}],
                "auth_credentials": {
                    "mode": "encrypted",
                    "encrypted_domain_url": "bauth-v1:encrypted-domain",
                    "encrypted_username": "bauth-v1:encrypted-username",
                    "encrypted_password": "bauth-v1:encrypted-password",
                },
                "task_execution_overrides_path": "old-overrides.json",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class LocalCliResolutionTests(unittest.TestCase):
    def test_typescript_cli_uses_checkout_local_tsx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "agent"
            cli = root / "src" / "cli.ts"
            tsx_loader = root / "node_modules" / "tsx" / "dist" / "loader.mjs"
            _write_executable(cli, "")
            _write_executable(tsx_loader, "")

            with patch(
                "rbbench.local_browser_agent.shutil.which",
                return_value="/test/bin/node",
            ):
                command = resolve_local_cli_command(cli)

            self.assertEqual(
                command,
                (
                    "/test/bin/node",
                    "--import",
                    str(tsx_loader.resolve()),
                    str(cli.resolve()),
                ),
            )

    def test_javascript_cli_uses_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cli = Path(temporary) / "dist" / "cli.js"
            _write_executable(cli, "")

            with patch(
                "rbbench.local_browser_agent.shutil.which",
                return_value="/test/bin/node",
            ):
                command = resolve_local_cli_command(cli)

            self.assertEqual(command, ("/test/bin/node", str(cli.resolve())))

    def test_native_cli_executes_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cli = Path(temporary) / "browser-agent"
            _write_executable(cli, "#!/bin/sh\n")

            self.assertEqual(resolve_local_cli_command(cli), (str(cli.resolve()),))


class LocalConfigOverlayTests(unittest.TestCase):
    def test_overlay_preserves_agent_config_and_replaces_attempt_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = root / "browser-agent"
            _write_executable(cli, "#!/bin/sh\nexit 0\n")
            config_path = root / "browser-agent.yaml"
            _base_config(config_path)
            relative_chrome = root / "bin" / "chrome"
            relative_chrome.parent.mkdir()
            relative_chrome.write_text("", encoding="utf-8")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["executable_path"] = "bin/chrome"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            attempt_dir = root / "attempt"
            artifacts = attempt_dir / "artifacts"
            profile = root / "fixture-profile"
            profile.mkdir()
            attempt = AttemptDescriptor(
                "attempt-local",
                "RBA-015",
                "https://example.test/start",
                attempt_dir,
                artifacts,
                session={
                    "user_data_dir": str(profile),
                    "credentials": {
                        "username": "fixture-user",
                        "password": "fixture-secret",
                    },
                },
            )
            executor = LocalBrowserAgentExecutor(
                cli_path=cli, config_path=config_path
            )

            generated, steps = executor._materialize_config(
                instruction="benchmark task",
                attempt=attempt,
                runtime_root=attempt_dir / "runtime",
            )
            resolved = yaml.safe_load(generated.read_text(encoding="utf-8"))

            self.assertEqual(resolved["stage_llms"], config["stage_llms"])
            self.assertEqual(resolved["feature_flags"], config["feature_flags"])
            self.assertTrue(resolved["headless"])
            self.assertEqual(resolved["max_steps"], 37)
            self.assertEqual(resolved["task_run_retry_count"], 2)
            self.assertEqual(resolved["tasks"], [{
                "task": "benchmark task",
                "url": "https://example.test/start",
            }])
            self.assertEqual(resolved["concurrency"], 1)
            self.assertEqual(resolved["task_runs"], 1)
            self.assertEqual(resolved["step_messages_jsonl_path"], str(steps))
            self.assertEqual(
                resolved["executable_path"], str(relative_chrome.resolve())
            )
            self.assertEqual(
                resolved["browser_profiles"]["seed_user_data_dir"],
                str(profile.resolve()),
            )
            self.assertEqual(
                resolved["auth_credentials"],
                {
                    "mode": "encrypted",
                    "encrypted_domain_url": "bauth-v1:encrypted-domain",
                    "encrypted_username": "bauth-v1:encrypted-username",
                    "encrypted_password": "bauth-v1:encrypted-password",
                },
            )
            self.assertNotIn("task_execution_overrides_path", resolved)
            self.assertNotIn("fixture-secret", generated.read_text(encoding="utf-8"))


class LocalBrowserAgentExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_login_preflight_runs_once_before_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = root / "fake-browser-agent"
            _write_executable(
                cli,
                f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

if sys.argv[1:] == ["--version-json"]:
    print(json.dumps({{"version": "local-test", "rpcProtocolVersion": 1}}))
elif sys.argv[1:] == ["codex-login", "--check"]:
    with Path(os.environ["LOGIN_COUNT"]).open("a", encoding="utf-8") as output:
        output.write("1\\n")
    print(json.dumps({{"loggedIn": True}}))
else:
    raise SystemExit(2)
""",
            )
            config_path = root / "browser-agent.yaml"
            _base_config(config_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["stage_llms"]["runAgent"]["provider"] = "codex"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            count = root / "login-count"
            executor = LocalBrowserAgentExecutor(
                cli_path=cli,
                config_path=config_path,
                timeout_seconds=10,
            )
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"LOGIN_COUNT": str(count)}),
                contextlib.redirect_stderr(stderr),
            ):
                await executor.prepare_run()

            self.assertEqual(count.read_text(encoding="utf-8"), "1\n")
            self.assertEqual(stderr.getvalue(), "")

    async def test_codex_login_check_rejects_logged_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = root / "fake-browser-agent"
            _write_executable(
                cli,
                f"""#!{sys.executable}
import json
import sys

if sys.argv[1:] == ["--version-json"]:
    print(json.dumps({{"version": "local-test", "rpcProtocolVersion": 1}}))
elif sys.argv[1:] == ["codex-login", "--check"]:
    print(json.dumps({{"loggedIn": False}}))
else:
    raise SystemExit(2)
""",
            )
            config_path = root / "browser-agent.yaml"
            _base_config(config_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["stage_llms"]["runAgent"]["provider"] = "codex"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(
                ExecutorError,
                r"Codex login is required\. Run 'rbbench codex-login' first, then retry\.",
            ):
                await LocalBrowserAgentExecutor(
                    cli_path=cli, config_path=config_path
                ).prepare_run()

    async def test_non_codex_preflight_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = root / "browser-agent"
            _write_executable(cli, "#!/bin/sh\nexit 99\n")
            config_path = root / "browser-agent.yaml"
            _base_config(config_path)

            await LocalBrowserAgentExecutor(
                cli_path=cli, config_path=config_path
            ).prepare_run()

    async def test_rpc_execution_uses_local_config_and_returns_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = root / "fake-browser-agent"
            _write_executable(
                cli,
                f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
import yaml

if sys.argv[1:] == [\"--version-json\"]:
    print(json.dumps({{\"version\": \"local-test\", \"rpcProtocolVersion\": 1}}))
    raise SystemExit(0)

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding=\"utf-8\"))
print(os.environ[\"TMPDIR\"], file=sys.stderr)
request = json.loads(sys.stdin.readline())
credentials = request[\"params\"][\"tasks\"][0][\"credentials\"]
assert credentials[0][\"password\"] == \"fixture-secret\"
trajectory = Path(config[\"step_messages_jsonl_path\"])
trajectory.parent.mkdir(parents=True, exist_ok=True)
trajectory.write_text(\"{{}}\\n\", encoding=\"utf-8\")
usage = trajectory.parent / \"tokenUsage\" / \"task-001.json\"
usage.parent.mkdir(parents=True, exist_ok=True)
usage.write_text(json.dumps({{\"schemaVersion\": 1, \"attempts\": [], \"totals\": {{}}}}), encoding=\"utf-8\")
print(json.dumps({{\"jsonrpc\": \"2.0\", \"id\": 1, \"result\": {{\"accepted\": True}}}}))
print(json.dumps({{\"jsonrpc\": \"2.0\", \"method\": \"browser-agent/task_result\", \"params\": {{\"task_id\": \"task-1\", \"status\": \"completed\", \"runs\": [{{\"completed\": True, \"yaml_result\": \"answer: ok\"}}]}}}}))
print(json.dumps({{\"jsonrpc\": \"2.0\", \"method\": \"browser-agent/all_tasks_completed\", \"params\": {{}}}}))
""",
            )
            config_path = root / "browser-agent.yaml"
            _base_config(config_path)
            attempt_dir = root / "attempt"
            artifacts = attempt_dir / "artifacts"
            attempt = AttemptDescriptor(
                "attempt-rpc",
                "RBA-046",
                "https://example.test/login",
                attempt_dir,
                artifacts,
                session={
                    "credentials": {
                        "username": "fixture-user",
                        "password": "fixture-secret",
                    }
                },
            )
            projected = types.SimpleNamespace(
                final_result="answer: ok",
                steps=["step one"],
                num_steps=1,
                duration_seconds=1.5,
                input_tokens=100,
                cached_input_tokens=80,
                output_tokens=50,
                reasoning_tokens=40,
                non_reasoning_output_tokens=10,
                total_tokens=150,
                model_invocations=1,
            )
            executor = LocalBrowserAgentExecutor(
                cli_path=cli,
                config_path=config_path,
                timeout_seconds=10,
            )

            live_stderr = io.StringIO()
            with (
                patch(
                    "rbbench.local_browser_agent.convert_trajectory",
                    return_value=projected,
                ),
                contextlib.redirect_stderr(live_stderr),
            ):
                result = await executor.execute(
                    load_catalog().by_id("RBA-046"), attempt
                )

            self.assertEqual(result.final_result, "answer: ok")
            self.assertEqual(result.steps, ["step one"])
            self.assertEqual(result.metrics["steps"], 1)
            self.assertEqual(result.metrics["input_tokens"], 100)
            self.assertEqual(result.metrics["cached_input_tokens"], 80)
            self.assertEqual(result.metrics["output_tokens"], 50)
            self.assertEqual(result.metrics["reasoning_tokens"], 40)
            self.assertEqual(result.metrics["cli_version"], "local-test")
            effective = artifacts / "browser-agent" / "effective-config.yaml"
            self.assertTrue(effective.is_file())
            self.assertNotIn(
                "fixture-secret", effective.read_text(encoding="utf-8")
            )
            self.assertTrue(
                (artifacts / "browser-agent" / "local-cli-provenance.json").is_file()
            )
            agent_tmpdir = Path(
                (artifacts / "browser-agent" / "cli-stderr.log")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(agent_tmpdir.name.startswith("rbbench-"))
            self.assertEqual(live_stderr.getvalue().strip(), str(agent_tmpdir))
            socket_path = agent_tmpdir / "tsx-1000" / "12345.pipe"
            self.assertLess(len(str(socket_path)), 108)
            self.assertFalse(agent_tmpdir.exists())


class LocalCliArgumentTests(unittest.TestCase):
    def test_local_cli_and_config_must_be_provided_together(self) -> None:
        args = build_parser().parse_args(
            ["run", "--name", "test", "--agent-cli", "src/cli.ts"]
        )
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            _executor(args)

    def test_local_cli_requires_browser_agent_executor(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--name",
                "test",
                "--executor",
                "dry-run",
                "--agent-cli",
                "src/cli.ts",
                "--agent-config",
                "agent.yaml",
            ]
        )
        with self.assertRaisesRegex(ValueError, "require --executor browser-agent"):
            _executor(args)


if __name__ == "__main__":
    unittest.main()
