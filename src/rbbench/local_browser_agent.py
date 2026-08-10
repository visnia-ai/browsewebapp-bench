from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import shutil
import signal
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from .browser_agent_artifacts import (
    convert_trajectory,
    load_token_usage,
    load_trajectory,
)
from .errors import ExecutorError
from .executors import (
    Executor,
    _browser_agent_instruction,
    _credential_records,
    _json_object_from_text,
    _show_browser_agent_log,
    _stage_inputs,
)
from .io import write_json
from .schema import AttemptDescriptor, ExecutionResult, TaskSpec


RPC_PROTOCOL_VERSION = 1
_DEBUG_PORT_MIN = 20_000
_DEBUG_PORT_MAX = 60_000


def _config_uses_codex(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("provider") == "codex":
            return True
        return any(_config_uses_codex(item) for item in value.values())
    if isinstance(value, list):
        return any(_config_uses_codex(item) for item in value)
    return False


def resolve_local_cli_command(cli_path: Path) -> tuple[str, ...]:
    cli = cli_path.expanduser().resolve()
    if not cli.is_file():
        raise ValueError(f"Local Browser Agent CLI does not exist: {cli}")

    if cli.suffix in {".ts", ".mts", ".cts"}:
        node = shutil.which("node")
        if not node:
            raise ValueError("Node.js is required to run a local TypeScript Browser Agent CLI.")
        for parent in cli.parents:
            tsx_loader = parent / "node_modules" / "tsx" / "dist" / "loader.mjs"
            if tsx_loader.is_file():
                return (node, "--import", str(tsx_loader.resolve()), str(cli))
        raise ValueError(
            f"Unable to run {cli} without a build: install its local dependencies "
            "so the tsx loader is available."
        )

    if cli.suffix in {".js", ".mjs", ".cjs"}:
        node = shutil.which("node")
        if not node:
            raise ValueError("Node.js is required to run a local Browser Agent CLI.")
        return (node, str(cli))

    if os.name != "nt" and not os.access(cli, os.X_OK):
        raise ValueError(f"Local Browser Agent CLI is not executable: {cli}")
    return (str(cli),)


class _DebugPortPool:
    """Keep concurrent local CLI processes in this benchmark on unique ports."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._leased: set[int] = set()

    @staticmethod
    def _available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    async def acquire(self) -> int:
        async with self._lock:
            for port in range(_DEBUG_PORT_MIN, _DEBUG_PORT_MAX + 1):
                if port in self._leased or not self._available(port):
                    continue
                self._leased.add(port)
                return port
        raise ExecutorError("No local Browser Agent debug port is available")

    async def release(self, port: int) -> None:
        async with self._lock:
            self._leased.discard(port)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        await process.wait()


async def _communicate_with_live_stderr(
    process: asyncio.subprocess.Process, stdin: bytes
) -> tuple[bytes, bytes]:
    """Communicate with a CLI while teeing its diagnostics."""
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    async def relay_stderr() -> bytes:
        chunks: list[bytes] = []
        while line := await process.stderr.readline():
            chunks.append(line)
            _show_browser_agent_log(line.decode(errors="replace").rstrip("\r\n"))
        return b"".join(chunks)

    try:
        process.stdin.write(stdin)
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        process.stdin.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()

    stdout_task = asyncio.create_task(process.stdout.read())
    stderr_task = asyncio.create_task(relay_stderr())
    try:
        await process.wait()
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        return stdout, stderr
    except BaseException:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise


def _rpc_task_result(stdout: bytes) -> dict[str, Any]:
    task_result: dict[str, Any] | None = None
    accepted = False
    completed = False
    errors: list[str] = []

    for line_number, raw_line in enumerate(stdout.decode(errors="replace").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ExecutorError(
                f"Local Browser Agent emitted invalid RPC JSON on line {line_number}"
            ) from exc
        if not isinstance(message, dict):
            raise ExecutorError("Local Browser Agent emitted a non-object RPC message")
        if message.get("id") == 1:
            if isinstance(message.get("error"), dict):
                error = message["error"]
                raise ExecutorError(
                    f"Local Browser Agent rejected the run: {error.get('message', 'unknown error')}"
                )
            accepted = message.get("result") == {"accepted": True}
        method = message.get("method")
        params = message.get("params")
        if method == "browser-agent/task_result" and isinstance(params, dict):
            task_result = params
        elif method == "browser-agent/error" and isinstance(params, dict):
            errors.append(str(params.get("message") or "Browser Agent RPC error"))
        elif method == "browser-agent/all_tasks_completed":
            completed = True

    if errors:
        raise ExecutorError("; ".join(errors))
    if not accepted:
        raise ExecutorError("Local Browser Agent did not accept the RPC run")
    if task_result is None:
        raise ExecutorError("Local Browser Agent returned no task result")
    if not completed:
        raise ExecutorError("Local Browser Agent did not complete its RPC lifecycle")
    return task_result


def _result_text(run: dict[str, Any]) -> str:
    yaml_result = run.get("yaml_result")
    if isinstance(yaml_result, str):
        return yaml_result
    data = run.get("data")
    if isinstance(data, str):
        return data
    if data is None:
        return ""
    return json.dumps(data, ensure_ascii=False)


class LocalBrowserAgentExecutor(Executor):
    """Run an ordinary Browser Agent config through a local CLI over stdio RPC."""

    def __init__(
        self,
        *,
        cli_path: Path,
        config_path: Path,
        timeout_seconds: int = 1800,
    ) -> None:
        self.cli_path = cli_path.expanduser().resolve()
        self.config_path = config_path.expanduser().resolve()
        self.command = resolve_local_cli_command(self.cli_path)
        if not self.config_path.is_file():
            raise ValueError(f"Browser Agent config does not exist: {self.config_path}")
        try:
            loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Unable to load Browser Agent config: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("Browser Agent config must contain a YAML mapping")
        self.base_config = loaded
        self.base_config_sha256 = hashlib.sha256(
            self.config_path.read_bytes()
        ).hexdigest()
        self.timeout_seconds = timeout_seconds
        self._version: dict[str, Any] | None = None
        self._version_lock = asyncio.Lock()
        self._ports = _DebugPortPool()

    async def prepare_run(self) -> None:
        if not _config_uses_codex(self.base_config):
            return
        await self._cli_version()
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                "codex-login",
                "--check",
                cwd=self.config_path.parent,
                env=os.environ.copy(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise ExecutorError(
                "Codex login check could not start Browser Agent"
            ) from exc
        try:
            stdout, _ = await asyncio.wait_for(
                _communicate_with_live_stderr(process, b""),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process(process)
            raise ExecutorError(
                f"Codex login check timed out after {self.timeout_seconds}s"
            ) from exc
        if process.returncode != 0:
            raise ExecutorError(
                f"Codex login check failed with exit code {process.returncode}"
            )
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutorError("Codex login check returned invalid output") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("loggedIn"), bool
        ):
            raise ExecutorError("Codex login check returned invalid output")
        if not payload["loggedIn"]:
            raise ExecutorError(
                "Codex login is required. Run 'rbbench codex-login' first, "
                "then retry."
            )

    async def _cli_version(self) -> dict[str, Any]:
        async with self._version_lock:
            if self._version is not None:
                return self._version
            process = await asyncio.create_subprocess_exec(
                *self.command,
                "--version-json",
                cwd=self.config_path.parent,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process(process)
                raise ExecutorError(
                    "Local Browser Agent version probe timed out"
                ) from exc
            if process.returncode != 0:
                detail = stderr.decode(errors="replace").strip()
                raise ExecutorError(
                    f"Local Browser Agent version probe failed: {detail or process.returncode}"
                )
            try:
                version = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise ExecutorError(
                    "Local Browser Agent returned invalid --version-json output"
                ) from exc
            if (
                not isinstance(version, dict)
                or version.get("rpcProtocolVersion") != RPC_PROTOCOL_VERSION
            ):
                raise ExecutorError(
                    "Local Browser Agent does not support RPC protocol version 1"
                )
            self._version = version
            return version

    def _materialize_config(
        self,
        *,
        instruction: str,
        attempt: AttemptDescriptor,
        runtime_root: Path,
    ) -> tuple[Path, Path]:
        config = copy.deepcopy(self.base_config)
        for key in (
            "task",
            "tasks",
            "browser_profiles",
            "browserProfiles",
            "concurrency",
            "download_dir",
            "downloadDir",
            "file_workspace_root",
            "fileWorkspaceRoot",
            "save_steps_context",
            "saveStepsContext",
            "save_task_logs",
            "saveTaskLogs",
            "step_messages_jsonl_path",
            "stepMessagesJsonlPath",
            "task_execution_overrides_path",
            "taskExecutionOverridesPath",
            "task_runs",
            "taskRuns",
            "wait_between_tasks_ms",
            "waitBetweenTasksMs",
        ):
            config.pop(key, None)

        executable = config.get("executable_path", config.get("executablePath"))
        if isinstance(executable, str) and executable and not Path(executable).is_absolute():
            config.pop("executablePath", None)
            config["executable_path"] = str(
                (self.config_path.parent / executable).resolve()
            )

        browser_agent_root = attempt.artifact_dir / "browser-agent"
        steps_path = browser_agent_root / "steps.jsonl"
        config.update(
            {
                "tasks": [{"task": instruction, "url": attempt.start_url}],
                "concurrency": 1,
                "task_runs": 1,
                "wait_between_tasks_ms": 0,
                "download_dir": str(attempt.artifact_dir / "downloads"),
                "file_workspace_root": str(attempt.artifact_dir),
                "save_steps_context": True,
                "save_task_logs": True,
                "step_messages_jsonl_path": str(steps_path),
            }
        )
        seed_profile = attempt.session.get("user_data_dir")
        if seed_profile:
            config["browser_profiles"] = {
                "mode": "seeded",
                "seed_user_data_dir": str(Path(str(seed_profile)).resolve()),
                "per_worker_user_data_root": str(runtime_root / "browser-profiles"),
                "reuse_existing_worker_profiles": False,
            }

        runtime_root.mkdir(parents=True, exist_ok=True)
        attempt.artifact_dir.mkdir(parents=True, exist_ok=True)
        (attempt.artifact_dir / "downloads").mkdir(parents=True, exist_ok=True)
        generated = runtime_root / "config.yaml"
        generated.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        generated.chmod(0o600)
        return generated, steps_path

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        version = await self._cli_version()
        resolved = dict(
            attempt.environment_data.get("resolved_fixture_artifacts", {})
        )
        staged = _stage_inputs(resolved, attempt.artifact_dir)
        instruction = _browser_agent_instruction(task, attempt, staged)
        runtime_root = attempt.attempt_dir / "browser-agent-local-runtime"
        generated_config, trajectory_path = self._materialize_config(
            instruction=instruction,
            attempt=attempt,
            runtime_root=runtime_root,
        )
        credentials = _credential_records(
            attempt.session.get("credentials"), attempt.start_url
        )
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "browser-agent/run",
            "params": {
                "tasks": [
                    {"credentials": credentials} if credentials else {}
                ]
            },
        }
        port = await self._ports.acquire()
        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix="rbbench-",
                dir="/tmp" if os.name != "nt" else None,
            )
        )
        environment = os.environ.copy()
        environment.update(
            {
                "TMPDIR": str(temporary_dir),
                "BROWSER_AGENT_DEBUG_PORT_MIN": str(port),
                "BROWSER_AGENT_DEBUG_PORT_MAX": str(port),
            }
        )
        started = time.perf_counter()
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                str(generated_config),
                "--rpc",
                cwd=self.config_path.parent,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_with_live_stderr(
                        process, (json.dumps(request) + "\n").encode()
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process(process)
                raise ExecutorError(
                    f"Local Browser Agent timed out after {self.timeout_seconds}s"
                ) from exc
        finally:
            await self._ports.release(port)
            shutil.rmtree(temporary_dir, ignore_errors=True)

        artifact_root = attempt.artifact_dir / "browser-agent"
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "cli-stderr.log").write_bytes(stderr)
        (artifact_root / "effective-config.yaml").write_bytes(
            generated_config.read_bytes()
        )
        write_json(
            artifact_root / "local-cli-provenance.json",
            {
                "cli_path": str(self.cli_path),
                "cli_command": list(self.command),
                "cli_version": version,
                "source_config": str(self.config_path),
                "source_config_sha256": self.base_config_sha256,
                "debug_port": port,
            },
        )

        if process is None or process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise ExecutorError(
                f"Local Browser Agent exited {process.returncode if process else 'before launch'}: "
                f"{detail or 'no diagnostic output'}"
            )
        task_result = _rpc_task_result(stdout)
        errors = [str(item) for item in task_result.get("errors", []) if item]
        runs = task_result.get("runs")
        if not isinstance(runs, list):
            raise ExecutorError(
                f"Local Browser Agent failed: {'; '.join(errors) or 'task did not complete'}"
            )
        completed_runs = [
            run for run in runs if isinstance(run, dict) and run.get("completed") is True
        ]
        final_result = _result_text(completed_runs[-1]) if completed_runs else ""

        try:
            usage_path = trajectory_path.parent / "tokenUsage" / "task-001.json"
            projected = convert_trajectory(
                load_trajectory(trajectory_path),
                expected_task=instruction,
                usage_totals=load_token_usage(usage_path),
            )
        except (OSError, ValueError) as exc:
            raise ExecutorError(
                f"Browser Agent trajectory or token usage is invalid: {exc}"
            ) from exc
        final_result = projected.final_result or final_result

        screenshots = sorted(
            str(path)
            for path in attempt.artifact_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        parsed = _json_object_from_text(final_result)
        observation = parsed or {
            "result": {"final_result": final_result},
            "safety": {},
        }
        observation.setdefault("safety", {})
        elapsed = time.perf_counter() - started
        return ExecutionResult(
            final_result=final_result or "Agent did not return a final result",
            steps=projected.steps,
            screenshots=screenshots,
            observation={**observation, "page": {"url": attempt.start_url}},
            metrics={
                "steps": projected.num_steps,
                "duration_seconds": (
                    projected.duration_seconds
                    if projected.duration_seconds > 0
                    else elapsed
                ),
                "cost": 0.0,
                "input_tokens": projected.input_tokens,
                "cached_input_tokens": projected.cached_input_tokens,
                "output_tokens": projected.output_tokens,
                "reasoning_tokens": projected.reasoning_tokens,
                "non_reasoning_output_tokens": projected.non_reasoning_output_tokens,
                "total_tokens": projected.total_tokens,
                "model_invocations": projected.model_invocations,
                "postprocessor": "native-browser-agent-bu-projection-v1",
                "source_trajectory": str(trajectory_path),
                "local_cli": str(self.cli_path),
                "cli_version": version.get("version"),
                "browser_agent_rpc_status": task_result.get("status"),
                "browser_agent_rpc_errors": errors,
            },
        )
