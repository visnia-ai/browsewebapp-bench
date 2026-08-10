from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .catalog import REPO_ROOT
from .environments import hook_variable
from .integrations import adapter_for
from .schema import TaskSpec

_SDK_PACKAGE = "browser-agent-python-sdk"
_PYPI_URL = f"https://pypi.org/pypi/{_SDK_PACKAGE}/json"


def _command_available(command: str) -> bool:
    argv = shlex.split(command)
    if not argv:
        return False
    executable = Path(argv[0])
    if executable.is_absolute() or "/" in argv[0]:
        return executable.exists()
    from shutil import which

    return which(argv[0]) is not None


def _latest_sdk_version() -> str | None:
    try:
        with urllib.request.urlopen(_PYPI_URL, timeout=5) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None
    version = payload.get("info", {}).get("version")
    return str(version) if version else None


def _version_key(version: str) -> tuple[object, ...]:
    parts: list[object] = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(part)
    return tuple(parts)


def _sdk_executor_issues() -> list[str]:
    if importlib.util.find_spec("browser_agent") is None:
        return [f"Install {_SDK_PACKAGE} (`pip install -U {_SDK_PACKAGE}`)"]
    try:
        installed = importlib.metadata.version(_SDK_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return [f"Install {_SDK_PACKAGE} (`pip install -U {_SDK_PACKAGE}`)"]
    latest = _latest_sdk_version()
    if latest is not None and _version_key(installed) < _version_key(latest):
        return [
            f"{_SDK_PACKAGE} {installed} is installed; upgrade to {latest} "
            f"(`pip install -U {_SDK_PACKAGE}`)"
        ]
    return []


def inspect_tasks(
    tasks: Iterable[TaskSpec],
    *,
    reference_dir: Path,
    executor: str | None = None,
    executor_command: str | None = None,
    judge: str | None = None,
    judge_command: str | None = None,
    judge_provider: str = "google",
    judge_api_key_env: str | None = None,
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for task in tasks:
        missing: list[str] = []
        warnings: list[str] = []
        if task.oracle.reference_key:
            reference = reference_dir / f"{task.oracle.reference_key}.json"
            if not reference.exists():
                warnings.append(
                    f"reference:{reference} (judge will score without ground truth)"
                )
        for key in ("input_artifact", "body_artifact", "content_artifact"):
            if key in task.fixture:
                fixture = REPO_ROOT / str(task.fixture[key])
                if not fixture.exists():
                    missing.append(f"fixture:{fixture}")
        if task.environment.adapter not in {"public_web", "ato_simulator"}:
            phases = {"prepare", "observe"}
            phases.update(task.environment.required_hooks)
            if task.environment.mutable:
                phases.add("cleanup")
            for phase in sorted(phases):
                variable = hook_variable(task.environment.adapter, phase)
                value = os.getenv(variable)
                if value and not _command_available(value):
                    missing.append(f"command:{variable}")
            if not all(
                os.getenv(hook_variable(task.environment.adapter, phase))
                for phase in phases
            ):
                missing.extend(adapter_for(task.environment.adapter).doctor(task.to_dict()))
        reports.append(
            {
                "task_id": task.task_id,
                "ready": not missing,
                "missing": missing,
                "warnings": warnings,
            }
        )

    executor_issues: list[str] = []
    if executor == "browser-agent":
        executor_issues.extend(_sdk_executor_issues())
    if executor == "browser-use" and importlib.util.find_spec("browser_use") is None:
        executor_issues.append("Install the browser-use optional dependency")
    if executor == "command":
        if not executor_command:
            executor_issues.append("--executor-command is required")
        elif not _command_available(executor_command):
            executor_issues.append("Executor command is not available")

    judge_issues: list[str] = []
    if judge == "llm":
        key_name = judge_api_key_env or {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }[judge_provider]
        if not os.getenv(key_name):
            judge_issues.append(f"Set {key_name} for the native LLM judge")
    if judge == "command":
        if not judge_command:
            judge_issues.append("--judge-command is required")
        elif not _command_available(judge_command):
            judge_issues.append("Judge command is not available")

    return {
        "ready": all(bool(item["ready"]) for item in reports)
        and not executor_issues
        and not judge_issues,
        "task_count": len(reports),
        "ready_task_count": sum(bool(item["ready"]) for item in reports),
        "executor_issues": executor_issues,
        "judge_issues": judge_issues,
        "tasks": reports,
    }
