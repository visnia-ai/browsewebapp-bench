from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .catalog import Catalog
from .environments import environment_for
from .errors import (
    BenchmarkError,
    CleanupError,
    ExecutorError,
    InvalidEnvironmentError,
    PolicyBlockedError,
)
from .executors import Executor
from .integrations.tally_provision import ensure_tally_forms
from .io import read_json, write_json
from .judges import Judge
from .schema import RunStatus, TaskRunResult, TaskSpec


class BenchmarkRunner:
    """Coordinates isolated attempts, trusted evidence capture, and LLM judgement.

    Agent harnesses are deliberately outside this class. Each attempt gets its own
    directory and subprocess/session, while environment adapters own setup,
    observation, and cleanup.
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        executor: Executor,
        judge: Judge,
        runtime_dir: Path,
        results_dir: Path,
        browser_profile: Path | None = None,
    ):
        self.catalog = catalog
        self.executor = executor
        self.judge = judge
        self.runtime_dir = runtime_dir
        self.results_dir = results_dir
        self.browser_profile = (
            browser_profile.resolve() if browser_profile is not None else None
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_one(
        self,
        task: TaskSpec,
        *,
        attempt_id: str | None = None,
        result_dir: Path | None = None,
    ) -> TaskRunResult:
        started = time.perf_counter()
        environment = environment_for(task, self.runtime_dir)
        attempt = environment.new_attempt(task, attempt_id)
        execution = None
        judgement = None
        status = RunStatus.AGENT_FAILURE
        error: str | None = None
        cleanup_error: str | None = None

        try:
            await environment.prepare(task, attempt)
            if self.browser_profile is not None:
                # CLI seed profile wins over any prepare-provided session profile.
                attempt.session["user_data_dir"] = str(self.browser_profile)
            execution = await self.executor.execute(task, attempt)
            if execution.error:
                raise ExecutorError(execution.error)
            observation = await environment.observe(task, attempt, execution)
            write_json(attempt.attempt_dir / "trusted-observation.json", observation)
            judgement = await self.judge.evaluate(
                task, attempt, execution, observation
            )
            status = RunStatus.SUCCESS if judgement.verdict else RunStatus.AGENT_FAILURE
            if not judgement.verdict:
                error = judgement.failure_reason or "LLM judge rejected the attempt"
        except PolicyBlockedError as exc:
            status = RunStatus.POLICY_BLOCK
            error = str(exc)
        except InvalidEnvironmentError as exc:
            status = RunStatus.INVALID_ENVIRONMENT
            error = str(exc)
        except (ExecutorError, BenchmarkError) as exc:
            status = RunStatus.AGENT_FAILURE
            error = str(exc)
        except Exception as exc:  # Preserve evidence and classify harness crashes.
            status = RunStatus.AGENT_FAILURE
            error = f"Unexpected {type(exc).__name__}: {exc}"
        finally:
            try:
                await environment.cleanup(task, attempt)
            except (CleanupError, InvalidEnvironmentError) as exc:
                cleanup_error = str(exc)
                status = RunStatus.INVALID_ENVIRONMENT
            except Exception as exc:
                cleanup_error = f"Unexpected {type(exc).__name__}: {exc}"
                status = RunStatus.INVALID_ENVIRONMENT

        result = TaskRunResult(
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            status=status,
            score=1.0 if judgement and judgement.verdict else 0.0,
            judgement=judgement,
            execution=execution,
            error=error,
            cleanup_error=cleanup_error,
            duration_seconds=time.perf_counter() - started,
        )
        write_json(attempt.attempt_dir / "result.json", result.to_dict())
        destination = result_dir or self.results_dir
        write_json(destination / f"{task.task_id}.json", result.to_dict())
        return result

    async def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        parallel: int = 1,
        run_id: str | None = None,
        resume: bool = False,
    ) -> dict[str, object]:
        selected = list(tasks)
        if parallel < 1:
            raise ValueError("parallel must be >= 1")
        identifier = run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        run_dir = self.results_dir / identifier
        run_dir.mkdir(parents=True, exist_ok=resume)

        completed: dict[str, TaskRunResult] = {}
        if resume:
            for task in selected:
                result_path = run_dir / f"{task.task_id}.json"
                if result_path.is_file():
                    completed[task.task_id] = TaskRunResult.from_dict(
                        read_json(result_path)
                    )
        pending = [task for task in selected if task.task_id not in completed]
        prepare_run = getattr(self.executor, "prepare_run", None)
        if pending and prepare_run is not None:
            await prepare_run()
        if any(task.environment.adapter == "tally_public_form" for task in pending):
            await asyncio.to_thread(ensure_tally_forms)

        global_gate = asyncio.Semaphore(parallel)
        key_limits: dict[str, int] = {}
        for task in selected:
            key = task.environment.concurrency_key
            key_limits[key] = min(
                key_limits.get(key, task.environment.concurrency_limit),
                task.environment.concurrency_limit,
            )
        domain_gates = {
            key: asyncio.Semaphore(limit) for key, limit in key_limits.items()
        }

        async def guarded(task: TaskSpec) -> TaskRunResult:
            # Acquire the narrower domain gate first. Acquiring the global gate
            # first lets several tasks for one concurrency-1 domain occupy every
            # global slot while merely waiting, starving independent domains.
            async with domain_gates[task.environment.concurrency_key]:
                async with global_gate:
                    attempt_base = f"{identifier}-{task.task_id.lower()}"
                    attempt_id = attempt_base
                    suffix = 2
                    while (self.runtime_dir / attempt_id).exists():
                        attempt_id = f"{attempt_base}-resume-{suffix}"
                        suffix += 1
                    return await self.run_one(
                        task,
                        attempt_id=attempt_id,
                        result_dir=run_dir,
                    )

        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        resumed_results = await asyncio.gather(*(guarded(task) for task in pending))
        results_by_task = {
            **completed,
            **{result.task_id: result for result in resumed_results},
        }
        results = [results_by_task[task.task_id] for task in selected]
        counts = Counter(result.status.value for result in results)
        summary: dict[str, object] = {
            "benchmark": self.catalog.name,
            "catalog_version": self.catalog.version,
            "run_id": identifier,
            "started_at": started_at,
            "duration_seconds": time.perf_counter() - started,
            "task_count": len(results),
            "status_counts": dict(counts),
            "mean_score": (
                sum(result.score for result in results) / len(results)
                if results
                else 0.0
            ),
            "results": [result.to_dict() for result in results],
        }
        write_json(run_dir / "summary.json", summary)
        return summary
