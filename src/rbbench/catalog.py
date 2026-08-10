from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import CatalogError
from .schema import TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "tasks" / "tasks.json"

BENCHMARK_TASK_IDS = tuple(f"RBA-{index:03d}" for index in range(1, 101))


@dataclass(frozen=True)
class Catalog:
    name: str
    version: str
    tasks: tuple[TaskSpec, ...]
    source_path: Path

    def by_id(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown task id: {task_id}")

    def select(self, task_ids: Iterable[str] | None = None) -> list[TaskSpec]:
        if task_ids is None:
            return list(self.tasks)
        return [self.by_id(task_id) for task_id in task_ids]

    def adapter_counts(self) -> Counter[str]:
        return Counter(task.environment.adapter for task in self.tasks)

    def validate(self) -> None:
        if len(self.tasks) != 100:
            raise CatalogError(
                f"Benchmark must contain 100 tasks, found {len(self.tasks)}"
            )
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            duplicates = sorted(task_id for task_id, n in Counter(ids).items() if n > 1)
            raise CatalogError(f"Duplicate task ids: {duplicates}")
        if ids != list(BENCHMARK_TASK_IDS):
            raise CatalogError("Task ids must be ordered RBA-001 through RBA-100")
        expected = {
            "ato_simulator": 9,
            "tally_public_form": 6,
            "public_web": 80,
            "controlled_portal": 5,
        }
        actual = dict(self.adapter_counts())
        if actual != expected:
            raise CatalogError(f"Unexpected benchmark environment distribution: {actual}")
        synthetic = [task for task in self.tasks if "synthetic" in task.environment.kind]
        if len(synthetic) != 5:
            raise CatalogError("Benchmark must contain exactly five controlled tasks")
        for task in self.tasks:
            if not task.sources:
                raise CatalogError(f"{task.task_id} has no source reference")
            if not task.environment.start_url.startswith("https://"):
                raise CatalogError(f"{task.task_id} start URL must use HTTPS")
            if any(not source.startswith("https://") for source in task.sources):
                raise CatalogError(f"{task.task_id} contains a non-HTTPS source")
            if task.environment.mutable and not task.cleanup.verify_absence:
                raise CatalogError(
                    f"Mutable task {task.task_id} must verify cleanup absence"
                )
            if task.environment.mutable:
                if "{{attempt_id}}" not in task.confirmed_task:
                    raise CatalogError(
                        f"Mutable task {task.task_id} must expose its attempt scope"
                    )
                required = {"prepare", "observe", "cleanup"}
                if not required.issubset(task.environment.required_hooks):
                    raise CatalogError(
                        f"Mutable task {task.task_id} requires prepare/observe/cleanup hooks"
                    )


def load_catalog(path: str | Path | None = None, *, strict: bool = True) -> Catalog:
    source = Path(path) if path else DEFAULT_CATALOG
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Catalog not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        raise CatalogError("Catalog root must contain a tasks array")
    profiles = raw.get("environment_profiles", {})
    if not isinstance(profiles, dict):
        raise CatalogError("environment_profiles must be an object")
    default_safety = raw.get("default_safety", {})
    if not isinstance(default_safety, dict):
        raise CatalogError("default_safety must be an object")
    oracle_profiles = raw.get("oracle_profiles", {})
    if not isinstance(oracle_profiles, dict):
        raise CatalogError("oracle_profiles must be an object")
    expanded: list[dict] = []
    for item in raw["tasks"]:
        if not isinstance(item, dict):
            raise CatalogError("Every task must be an object")
        task = dict(item)
        profile_name = task.pop("environment_profile", None)
        if profile_name:
            if profile_name not in profiles:
                raise CatalogError(f"Unknown environment profile: {profile_name}")
            profile = profiles[profile_name]
            if not isinstance(profile, dict):
                raise CatalogError(f"Environment profile must be an object: {profile_name}")
            task["environment"] = {
                **profile,
                **dict(task.get("environment", {})),
            }
        oracle_profile_name = task.pop("oracle_profile", None)
        if oracle_profile_name:
            if oracle_profile_name not in oracle_profiles:
                raise CatalogError(f"Unknown oracle profile: {oracle_profile_name}")
            oracle_profile = oracle_profiles[oracle_profile_name]
            if not isinstance(oracle_profile, dict):
                raise CatalogError(f"Oracle profile must be an object: {oracle_profile_name}")
            oracle_override = dict(task.get("oracle", {}))
            task["oracle"] = {
                **oracle_profile,
                **oracle_override,
                "assertions": oracle_override.get(
                    "assertions", oracle_profile.get("assertions", [])
                ),
            }
        task["safety"] = {**default_safety, **dict(task.get("safety", {}))}
        expanded.append(task)
    catalog = Catalog(
        name=str(raw.get("name", "unnamed")),
        version=str(raw.get("version", "unknown")),
        tasks=tuple(TaskSpec.from_dict(item) for item in expanded),
        source_path=source.resolve(),
    )
    if strict:
        catalog.validate()
    return catalog
