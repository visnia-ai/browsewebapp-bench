from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import CatalogError


class RunStatus(StrEnum):
    SUCCESS = "success"
    AGENT_FAILURE = "agent_failure"
    INVALID_ENVIRONMENT = "invalid_environment"
    POLICY_BLOCK = "policy_block"


@dataclass(frozen=True)
class EnvironmentSpec:
    adapter: str
    kind: str
    start_url: str
    auth: str = "none"
    mutable: bool = False
    concurrency_key: str = "default"
    concurrency_limit: int = 1
    required_hooks: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnvironmentSpec":
        required = ("adapter", "kind", "start_url")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise CatalogError(f"Environment is missing: {', '.join(missing)}")
        limit = int(raw.get("concurrency_limit", 1))
        if limit < 1:
            raise CatalogError("environment.concurrency_limit must be >= 1")
        return cls(
            adapter=str(raw["adapter"]),
            kind=str(raw["kind"]),
            start_url=str(raw["start_url"]),
            auth=str(raw.get("auth", "none")),
            mutable=bool(raw.get("mutable", False)),
            concurrency_key=str(raw.get("concurrency_key", raw["adapter"])),
            concurrency_limit=limit,
            required_hooks=tuple(str(x) for x in raw.get("required_hooks", [])),
        )


@dataclass(frozen=True)
class AssertionSpec:
    kind: str
    path: str = ""
    expected: Any = None
    expected_path: str | None = None
    required: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AssertionSpec":
        if not raw.get("kind"):
            raise CatalogError("Oracle assertion requires kind")
        return cls(
            kind=str(raw["kind"]),
            path=str(raw.get("path", "")),
            expected=raw.get("expected"),
            expected_path=(
                str(raw["expected_path"]) if raw.get("expected_path") else None
            ),
            required=bool(raw.get("required", True)),
            description=str(raw.get("description", "")),
        )


@dataclass(frozen=True)
class OracleSpec:
    type: str
    assertions: tuple[AssertionSpec, ...]
    reference_key: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OracleSpec":
        if not raw.get("type"):
            raise CatalogError("Oracle requires type")
        assertions = tuple(
            AssertionSpec.from_dict(item) for item in raw.get("assertions", [])
        )
        if not assertions:
            raise CatalogError("Oracle requires at least one assertion")
        return cls(
            type=str(raw["type"]),
            assertions=assertions,
            reference_key=(
                str(raw["reference_key"]) if raw.get("reference_key") else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleanupSpec:
    strategy: str
    verify_absence: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CleanupSpec":
        if not raw.get("strategy"):
            raise CatalogError("Cleanup requires strategy")
        return cls(
            strategy=str(raw["strategy"]),
            verify_absence=bool(raw.get("verify_absence", False)),
        )


@dataclass(frozen=True)
class SafetySpec:
    forbidden_actions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafetySpec":
        return cls(
            forbidden_actions=tuple(str(x) for x in raw.get("forbidden_actions", [])),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    confirmed_task: str
    category: str
    environment: EnvironmentSpec
    fixture: dict[str, Any]
    oracle: OracleSpec
    cleanup: CleanupSpec
    safety: SafetySpec
    sources: tuple[str, ...]
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpec":
        required = ("task_id", "title", "confirmed_task", "category")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise CatalogError(f"Task is missing: {', '.join(missing)}")
        task_id = str(raw["task_id"])
        if not task_id.startswith("RBA-"):
            raise CatalogError(f"Task id must start with RBA-: {task_id}")
        return cls(
            task_id=task_id,
            title=str(raw["title"]),
            confirmed_task=str(raw["confirmed_task"]),
            category=str(raw["category"]),
            environment=EnvironmentSpec.from_dict(dict(raw.get("environment", {}))),
            fixture=dict(raw.get("fixture", {})),
            oracle=OracleSpec.from_dict(dict(raw.get("oracle", {}))),
            cleanup=CleanupSpec.from_dict(dict(raw.get("cleanup", {}))),
            safety=SafetySpec.from_dict(dict(raw.get("safety", {}))),
            sources=tuple(str(x) for x in raw.get("sources", [])),
            tags=tuple(str(x) for x in raw.get("tags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptDescriptor:
    attempt_id: str
    task_id: str
    start_url: str
    attempt_dir: Path
    artifact_dir: Path
    session: dict[str, Any] = field(default_factory=dict)
    environment_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "start_url": self.start_url,
            "attempt_dir": str(self.attempt_dir),
            "artifact_dir": str(self.artifact_dir),
            "session": self.session,
            "environment_data": self.environment_data,
        }


@dataclass
class ExecutionResult:
    final_result: str = ""
    steps: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    observation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutionResult":
        return cls(
            final_result=str(raw.get("final_result", "")),
            steps=[str(x) for x in raw.get("steps", [])],
            screenshots=[str(x) for x in raw.get("screenshots", [])],
            observation=dict(raw.get("observation", {})),
            metrics=dict(raw.get("metrics", {})),
            error=(str(raw["error"]) if raw.get("error") else None),
        )


@dataclass(frozen=True)
class JudgementResult:
    reasoning: str
    verdict: bool
    failure_reason: str = ""
    impossible_task: bool = False
    reached_captcha: bool = False
    model: str | None = None
    provider: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JudgementResult":
        if "verdict" not in raw:
            raise ValueError("Judgement result requires verdict")
        return cls(
            reasoning=str(raw.get("reasoning", "")),
            verdict=bool(raw["verdict"]),
            failure_reason=str(raw.get("failure_reason", "")),
            impossible_task=bool(raw.get("impossible_task", False)),
            reached_captcha=bool(raw.get("reached_captcha", False)),
            model=(str(raw["model"]) if raw.get("model") else None),
            provider=(str(raw["provider"]) if raw.get("provider") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRunResult:
    task_id: str
    attempt_id: str
    status: RunStatus
    score: float
    judgement: JudgementResult | None = None
    execution: ExecutionResult | None = None
    error: str | None = None
    cleanup_error: str | None = None
    duration_seconds: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskRunResult":
        judgement = raw.get("judgement")
        execution = raw.get("execution")
        return cls(
            task_id=str(raw["task_id"]),
            attempt_id=str(raw["attempt_id"]),
            status=RunStatus(str(raw["status"])),
            score=float(raw.get("score", 0.0)),
            judgement=(
                JudgementResult.from_dict(dict(judgement)) if judgement else None
            ),
            execution=(
                ExecutionResult.from_dict(dict(execution)) if execution else None
            ),
            error=(str(raw["error"]) if raw.get("error") else None),
            cleanup_error=(
                str(raw["cleanup_error"]) if raw.get("cleanup_error") else None
            ),
            duration_seconds=float(raw.get("duration_seconds", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
