from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .errors import CleanupError, InvalidEnvironmentError, PolicyBlockedError
from .io import artifact_inventory, read_json, write_json
from .schema import AttemptDescriptor, ExecutionResult, TaskSpec
from .catalog import REPO_ROOT


ARTIFACT_FIXTURE_KEYS = {"input_artifact", "body_artifact", "content_artifact"}


def resolved_fixture_artifacts(task: TaskSpec) -> dict[str, str]:
    return {
        key: str((REPO_ROOT / str(value)).resolve())
        for key, value in task.fixture.items()
        if key in ARTIFACT_FIXTURE_KEYS
    }


def hook_variable(adapter: str, phase: str) -> str:
    normalized = adapter.upper().replace("-", "_")
    return f"RBBENCH_{normalized}_{phase.upper()}_CMD"


async def _run_hook(
    command: str,
    *,
    context_file: Path,
    output_file: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    argv = shlex.split(command)
    if not argv:
        raise InvalidEnvironmentError("Hook command is empty")
    env = os.environ.copy()
    env["RBBENCH_CONTEXT_FILE"] = str(context_file)
    env["RBBENCH_OUTPUT_FILE"] = str(output_file)
    process = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise InvalidEnvironmentError(
            f"Hook timed out after {timeout_seconds}s: {command}"
        ) from exc
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise InvalidEnvironmentError(
            f"Hook exited {process.returncode}: {message or command}"
        )
    if output_file.exists():
        return read_json(output_file)
    text = stdout.decode(errors="replace").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidEnvironmentError(
            f"Hook did not write output JSON or emit a JSON object: {command}"
        ) from exc
    if not isinstance(parsed, dict):
        raise InvalidEnvironmentError(f"Hook output must be a JSON object: {command}")
    return parsed


class EnvironmentAdapter(ABC):
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir

    def new_attempt(self, task: TaskSpec, attempt_id: str | None) -> AttemptDescriptor:
        identifier = attempt_id or uuid.uuid4().hex[:16]
        attempt_dir = self.runtime_dir / identifier
        artifact_dir = attempt_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        return AttemptDescriptor(
            attempt_id=identifier,
            task_id=task.task_id,
            start_url=task.environment.start_url,
            attempt_dir=attempt_dir,
            artifact_dir=artifact_dir,
        )

    @abstractmethod
    async def prepare(self, task: TaskSpec, attempt: AttemptDescriptor) -> None: ...

    @abstractmethod
    async def observe(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def cleanup(self, task: TaskSpec, attempt: AttemptDescriptor) -> None: ...


class PublicWebEnvironment(EnvironmentAdapter):
    async def prepare(self, task: TaskSpec, attempt: AttemptDescriptor) -> None:
        attempt.environment_data["resolved_fixture_artifacts"] = resolved_fixture_artifacts(task)
        write_json(attempt.attempt_dir / "task.json", task.to_dict())
        write_json(attempt.attempt_dir / "attempt.json", attempt.to_dict())

    async def observe(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
    ) -> dict[str, Any]:
        observation = dict(execution.observation)
        observation.setdefault("result", {"final_result": execution.final_result})
        observation["artifacts"] = artifact_inventory(attempt.artifact_dir)
        observation.setdefault("safety", {})
        return observation

    async def cleanup(self, task: TaskSpec, attempt: AttemptDescriptor) -> None:
        # Attempt artifacts are retained as benchmark evidence. Browser contexts are
        # owned and closed by the trusted executor.
        return None


class HookedEnvironment(EnvironmentAdapter):
    """Adapter for authenticated or mutable real environments.

    Hooks are deliberately external processes. Credentials remain in the operator's
    secret manager, while this repository owns the lifecycle contract and scoring.
    """

    def _command(self, task: TaskSpec, phase: str) -> str | None:
        override = os.environ.get(hook_variable(task.environment.adapter, phase))
        if override:
            return override
        return shlex.join(
            [
                sys.executable,
                "-m",
                "rbbench.integrations.hook",
                task.environment.adapter,
                phase,
            ]
        )

    def _context(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": task.to_dict(),
            "attempt": attempt.to_dict(),
        }
        if execution is not None:
            payload["execution"] = {
                "final_result": execution.final_result,
                "steps": execution.steps,
                "screenshots": execution.screenshots,
                "metrics": execution.metrics,
                "error": execution.error,
            }
        return payload

    def _require_hook(self, task: TaskSpec, phase: str) -> str:
        command = self._command(task, phase)
        if not command:
            variable = hook_variable(task.environment.adapter, phase)
            raise InvalidEnvironmentError(
                f"{task.task_id} requires {phase} hook; set {variable}"
            )
        return command

    async def prepare(self, task: TaskSpec, attempt: AttemptDescriptor) -> None:
        attempt.environment_data["resolved_fixture_artifacts"] = resolved_fixture_artifacts(task)
        task_file = attempt.attempt_dir / "task.json"
        attempt_file = attempt.attempt_dir / "attempt.json"
        write_json(task_file, task.to_dict())
        write_json(attempt_file, self._context(task, attempt))
        command = self._require_hook(task, "prepare")
        output = await _run_hook(
            command,
            context_file=attempt_file,
            output_file=attempt.attempt_dir / "prepare-output.json",
        )
        if output.get("policy_block"):
            raise PolicyBlockedError(str(output.get("reason", "Policy blocked")))
        attempt.session.update(dict(output.get("session", {})))
        attempt.environment_data.update(dict(output.get("environment_data", {})))
        if output.get("start_url"):
            attempt.start_url = str(output["start_url"])
        write_json(attempt_file, self._context(task, attempt))

    async def observe(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
    ) -> dict[str, Any]:
        context_file = attempt.attempt_dir / "observe-context.json"
        write_json(context_file, self._context(task, attempt, execution))
        command = self._require_hook(task, "observe")
        observation = await _run_hook(
            command,
            context_file=context_file,
            output_file=attempt.attempt_dir / "observation.json",
        )
        if observation.get("policy_block"):
            raise PolicyBlockedError(str(observation.get("reason", "Policy blocked")))
        observation["artifacts"] = artifact_inventory(attempt.artifact_dir)
        observation.setdefault("safety", {})
        return observation

    async def cleanup(self, task: TaskSpec, attempt: AttemptDescriptor) -> None:
        command = self._command(task, "cleanup")
        if not command:
            if task.environment.mutable or "cleanup" in task.environment.required_hooks:
                variable = hook_variable(task.environment.adapter, "cleanup")
                raise CleanupError(f"Cleanup hook missing; set {variable}")
            return
        context_file = attempt.attempt_dir / "cleanup-context.json"
        write_json(context_file, self._context(task, attempt))
        try:
            output = await _run_hook(
                command,
                context_file=context_file,
                output_file=attempt.attempt_dir / "cleanup-output.json",
            )
        except InvalidEnvironmentError as exc:
            raise CleanupError(str(exc)) from exc
        if task.cleanup.verify_absence and not output.get("absence_verified", False):
            raise CleanupError("Cleanup hook did not prove absence of attempt state")


def environment_for(task: TaskSpec, runtime_dir: Path) -> EnvironmentAdapter:
    if task.environment.adapter in {"public_web", "ato_simulator"}:
        return PublicWebEnvironment(runtime_dir)
    if task.environment.adapter in {
        "tally_public_form",
        "controlled_portal",
    }:
        return HookedEnvironment(runtime_dir)
    raise InvalidEnvironmentError(
        f"Unknown environment adapter: {task.environment.adapter}"
    )
