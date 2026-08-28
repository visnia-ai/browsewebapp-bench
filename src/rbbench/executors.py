from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import shlex
import shutil
import tempfile
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .browser_agent_artifacts import (
    convert_trajectory,
    load_token_usage,
    load_trajectory,
)
from .errors import ExecutorError
from .io import read_json, write_json
from .schema import AttemptDescriptor, ExecutionResult, TaskSpec


def _show_browser_agent_log(message: str) -> None:
    """Make interactive Browser Agent diagnostics visible to rbbench users."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


class _CompletionCreateProxy:
    def __init__(self, create, extra_body: dict[str, Any]):
        self._create = create
        self._extra_body = extra_body

    async def create(self, *args: Any, **kwargs: Any):
        merged = dict(kwargs.pop("extra_body", {}) or {})
        merged.update(self._extra_body)
        return await self._create(*args, extra_body=merged, **kwargs)


class _OpenAIClientProxy:
    """Inject provider-routing fields unsupported by Browser Use 0.11.x."""

    def __init__(self, client: Any, extra_body: dict[str, Any]):
        self._client = client
        self.chat = type("ChatProxy", (), {})()
        self.chat.completions = type("CompletionsProxy", (), {})()
        self.chat.completions.create = _CompletionCreateProxy(
            client.chat.completions.create, extra_body
        ).create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


_CHROME_PROFILE_TRANSIENT_NAMES = {
    "DevToolsActivePort",
    "SingletonCookie",
    "SingletonLock",
    "SingletonSocket",
}


def _clone_chrome_profile(source: Path, *, prefix: str) -> Path:
    """Copy an authenticated Chrome seed without carrying live-process locks."""
    if not source.is_dir():
        raise ExecutorError(f"Chrome profile seed is not a directory: {source}")
    destination = Path(tempfile.mkdtemp(prefix=prefix))

    def ignore_transient(_directory: str, names: list[str]) -> set[str]:
        return _CHROME_PROFILE_TRANSIENT_NAMES.intersection(names)

    try:
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=ignore_transient,
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


async def _start_browser_at(browser: Any, url: str) -> None:
    """Start Browser Use on the exact prepared URL, including fixture markers."""
    await browser.start()
    await browser.navigate_to(url)


def _install_openai_usage_tracker(llm: Any) -> dict[str, int]:
    """Capture OpenAI's reasoning-token subset without storing reasoning text."""
    totals = {"reasoning_tokens": 0, "responses": 0}
    original = getattr(llm, "_get_usage", None)
    if not callable(original):
        return totals

    def tracked(response: Any):
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0)
        if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool):
            totals["reasoning_tokens"] += max(0, int(reasoning))
        if usage is not None:
            totals["responses"] += 1
        return original(response)

    object.__setattr__(llm, "_get_usage", tracked)
    return totals


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class Executor(ABC):
    async def prepare_run(self) -> None:
        """Perform run-wide setup before any parallel attempts are scheduled."""
        return None

    @abstractmethod
    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult: ...


class DryRunExecutor(Executor):
    """Writes no target state; useful for lifecycle and failure-path testing."""

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        return ExecutionResult(
            final_result="Dry run: no browser actions were performed.",
            observation={
                "result": {"dry_run": True},
                "page": {"url": attempt.start_url, "text": ""},
                "state": {},
                "safety": {"forbidden_action_performed": False},
            },
            metrics={"steps": 0, "duration_seconds": 0.0, "cost": 0.0},
        )


class CommandExecutor(Executor):
    """Runs a trusted agent harness using a filesystem JSON contract."""

    def __init__(self, command: str, timeout_seconds: int = 1800):
        self.argv = shlex.split(command)
        if not self.argv:
            raise ValueError("Executor command is empty")
        self.timeout_seconds = timeout_seconds

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        task_file = attempt.attempt_dir / "executor-task.json"
        attempt_file = attempt.attempt_dir / "executor-attempt.json"
        output_file = attempt.attempt_dir / "executor-output.json"
        write_json(task_file, task.to_dict())
        write_json(attempt_file, attempt.to_dict())
        env = os.environ.copy()
        env.update(
            {
                "RBBENCH_TASK_FILE": str(task_file),
                "RBBENCH_ATTEMPT_FILE": str(attempt_file),
                "RBBENCH_OUTPUT_FILE": str(output_file),
                "RBBENCH_ARTIFACT_DIR": str(attempt.artifact_dir),
            }
        )
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await asyncio.shield(_terminate_command_process_tree(process))
            raise ExecutorError(
                f"Executor timed out after {self.timeout_seconds}s"
            ) from exc
        except asyncio.CancelledError:
            await asyncio.shield(_terminate_command_process_tree(process))
            raise
        if process.returncode != 0:
            message = stderr.decode(errors="replace").strip()
            raise ExecutorError(
                f"Executor exited {process.returncode}: {message or self.argv[0]}"
            )
        if output_file.exists():
            raw = read_json(output_file)
        else:
            text = stdout.decode(errors="replace").strip()
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ExecutorError(
                    "Executor did not write RBBENCH_OUTPUT_FILE or emit JSON"
                ) from exc
        result = ExecutionResult.from_dict(raw)
        result.metrics.setdefault("duration_seconds", time.perf_counter() - started)
        return result


async def _terminate_command_process_tree(
    process: asyncio.subprocess.Process,
) -> None:
    """Terminate a command adapter and every browser/agent child it spawned."""
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


def _stage_inputs(
    resolved: dict[str, Any], artifact_dir: Path
) -> dict[str, str]:
    staged: dict[str, str] = {}
    input_dir = artifact_dir / "inputs"
    for key, raw in resolved.items():
        source = Path(str(raw)).resolve()
        if not source.is_file():
            raise ExecutorError(f"Resolved input artifact does not exist: {source}")
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = input_dir / source.name
        shutil.copy2(source, destination)
        staged[str(key)] = f"./inputs/{destination.name}"
    return staged


def _browser_agent_instruction(
    task: TaskSpec, attempt: AttemptDescriptor, staged_inputs: dict[str, str]
) -> str:
    confirmed = task.confirmed_task.replace("{{attempt_id}}", attempt.attempt_id)
    forbidden = "\n".join(
        f"- {item}" for item in task.safety.forbidden_actions
    ) or "- none"
    model_fixture = {**task.fixture, **staged_inputs}
    return (
        f"Attempt ID: {attempt.attempt_id}\n"
        f"Start at {attempt.start_url}.\n\n"
        f"{confirmed}\n\n"
        f"Fixture data: {json.dumps(model_fixture, ensure_ascii=False)}\n"
        "Resolved input artifacts: "
        f"{json.dumps(staged_inputs, ensure_ascii=False)}\n\n"
        f"Forbidden actions:\n{forbidden}\n\n"
        "Use the browser UI to complete the task. When finished, return a concise "
        "answer containing every requested fact or an exact description of the "
        "completed action. Do not fabricate success or information."
    )


def _credential_records(raw: Any, default_domain: str) -> list[dict[str, str]]:
    if not raw:
        return []
    values = raw if isinstance(raw, list) else [raw]
    result: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ExecutorError("Session credentials must be an object or list")
        username = str(value.get("username") or value.get("email") or "")
        password = str(value.get("password") or "")
        domain = str(value.get("domain") or default_domain)
        if not username or not password or not domain:
            raise ExecutorError(
                "Each session credential requires username, password, and domain"
            )
        result.append(
            {"username": username, "password": password, "domain": domain}
        )
    return result


class BrowserAgentExecutor(Executor):
    """Default executor backed by the published Browser Agent Python SDK."""

    # Each published CLI process owns an in-process Chrome debug-port allocator.
    # Starting several SDK processes at the same instant can therefore race
    # between probing and binding the same port. Keep process starts slightly
    # apart while leaving the browser tasks themselves fully parallel.
    _SDK_LAUNCH_INTERVAL_SECONDS = 2.0

    def __init__(
        self,
        *,
        provider: str = "vllm",
        model: str = "nvidia/GLM-5.2-NVFP4",
        endpoint_url: str | None = None,
        reasoning_effort: str = "high",
        api_key: str | None = None,
        openrouter_provider: str | None = None,
        max_model_len: int = 48_000,
        reserve_output_tokens: int = 4_000,
        headless: bool = False,
        executable_path: str | None = None,
        max_steps: int = 50,
        retry_count: int = 0,
        timeout_seconds: int = 1800,
    ):
        self.provider = provider
        self.model = model
        self.endpoint_url = endpoint_url
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key
        self.openrouter_provider = openrouter_provider
        self.max_model_len = max_model_len
        self.reserve_output_tokens = reserve_output_tokens
        self.headless = headless
        self.executable_path = executable_path
        self.max_steps = max_steps
        self.retry_count = retry_count
        self.timeout_seconds = timeout_seconds
        self._sdk_launch_lock = asyncio.Lock()
        self._next_sdk_launch_at = 0.0

    async def _start_sdk_run(self, agent: Any, task: Any) -> Any:
        async with self._sdk_launch_lock:
            loop = asyncio.get_running_loop()
            delay = self._next_sdk_launch_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            run = agent.run(task)
            self._next_sdk_launch_at = (
                loop.time() + self._SDK_LAUNCH_INTERVAL_SECONDS
            )
            return run

    async def prepare_run(self) -> None:
        if self.provider != "codex":
            return
        try:
            from browser_agent import check_codex_login
        except ImportError as exc:
            raise ExecutorError(
                "Browser Agent SDK with Codex login checks is not installed; "
                "upgrade `browser-agent-python-sdk`"
            ) from exc
        try:
            logged_in = await check_codex_login(
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            raise ExecutorError(f"Codex login check failed: {exc}") from exc
        if not logged_in:
            raise ExecutorError(
                "Codex login is required. Run 'rbbench codex-login' first, "
                "then retry."
            )

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        try:
            from browser_agent import (
                BrowserAgent,
                BrowserAgentCredential,
                BrowserAgentTask,
            )
        except ImportError as exc:
            raise ExecutorError(
                "Browser Agent SDK is not installed; run "
                "`pip install -U browser-agent-python-sdk`"
            ) from exc

        if attempt.session.get("cdp_url") or attempt.session.get("storage_state"):
            raise ExecutorError(
                "The Browser Agent SDK runner requires a browser profile directory; "
                "CDP and storage-state sessions are not supported by this adapter"
            )

        resolved = dict(
            attempt.environment_data.get("resolved_fixture_artifacts", {})
        )
        staged = _stage_inputs(resolved, attempt.artifact_dir)
        instruction = _browser_agent_instruction(task, attempt, staged)
        downloads = attempt.artifact_dir / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        credentials = tuple(
            BrowserAgentCredential(**value)
            for value in _credential_records(
                attempt.session.get("credentials"), attempt.start_url
            )
        )
        options: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "download_directory": str(downloads),
            "reasoning_effort": self.reasoning_effort,
            "api_key": self.api_key,
            "endpoint_url": self.endpoint_url,
            "openrouter_provider": self.openrouter_provider,
            "max_model_len": self.max_model_len,
            "reserve_output_tokens": self.reserve_output_tokens,
            "headless": self.headless,
            "executable_path": self.executable_path,
            "workspace_directory": str(attempt.artifact_dir),
            "browser_profile_directory": attempt.session.get("user_data_dir"),
            "user_takeover_tool": False,
            "max_steps": self.max_steps,
            "concurrency": 1,
            "runs_per_task": 1,
            "retry_count": self.retry_count,
            "on_log": lambda entry: _show_browser_agent_log(entry.message),
        }
        agent = BrowserAgent(**options)
        run = await self._start_sdk_run(
            agent,
            BrowserAgentTask(
                task=instruction,
                url=attempt.start_url,
                credentials=credentials,
            ),
        )
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.shield(run.result), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await run.cancel()
            raise ExecutorError(
                f"Browser Agent timed out after {self.timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise ExecutorError(f"Browser Agent SDK failed: {exc}") from exc

        if not result.tasks:
            raise ExecutorError("Browser Agent returned no task result")
        task_result = result.tasks[0]
        completed_runs = [item for item in task_result.runs if item.completed]
        selected = completed_runs[-1] if completed_runs else (
            task_result.runs[-1] if task_result.runs else None
        )
        data = selected.data if selected is not None else None
        if isinstance(data, str):
            final_result = data
        elif data is None:
            final_result = ""
        else:
            final_result = json.dumps(data, ensure_ascii=False)

        trajectory_path = (
            attempt.artifact_dir / "browser-agent" / "steps.jsonl"
        )
        projected = None
        if trajectory_path.is_file():
            try:
                usage_path = trajectory_path.parent / "tokenUsage" / "task-001.json"
                projected = convert_trajectory(
                    load_trajectory(trajectory_path),
                    expected_task=instruction,
                    usage_totals=load_token_usage(usage_path),
                )
            except (OSError, ValueError) as exc:
                raise ExecutorError(
                    f"Browser Agent trajectory is invalid: {exc}"
                ) from exc
            final_result = projected.final_result
        errors = [item for item in task_result.errors if item]
        if (task_result.status != "completed" or selected is None) and projected is None:
            detail = "; ".join(errors) or "task did not complete"
            raise ExecutorError(f"Browser Agent failed: {detail}")

        screenshots = sorted(
            str(path)
            for path in attempt.artifact_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        elapsed = time.perf_counter() - started
        parsed = _json_object_from_text(final_result)
        observation = parsed or {
            "result": {"final_result": final_result},
            "safety": {},
        }
        observation.setdefault("safety", {})
        return ExecutionResult(
            final_result=final_result or "Agent did not return a final result",
            steps=projected.steps if projected is not None else [],
            screenshots=screenshots,
            observation={**observation, "page": {"url": attempt.start_url}},
            metrics={
                "steps": projected.num_steps if projected is not None else None,
                "duration_seconds": (
                    projected.duration_seconds
                    if projected is not None and projected.duration_seconds > 0
                    else elapsed
                ),
                "cost": 0.0,
                **(
                    {
                        "input_tokens": projected.input_tokens,
                        "cached_input_tokens": projected.cached_input_tokens,
                        "output_tokens": projected.output_tokens,
                        "reasoning_tokens": projected.reasoning_tokens,
                        "non_reasoning_output_tokens": projected.non_reasoning_output_tokens,
                        "total_tokens": projected.total_tokens,
                        "model_invocations": projected.model_invocations,
                    }
                    if projected is not None
                    else {}
                ),
                "sdk_run_id": result.run_id,
                "sdk_status": result.status,
                "browser_agent_task_status": task_result.status,
                "browser_agent_task_errors": errors,
                **(
                    {
                        "postprocessor": "native-browser-agent-bu-projection-v1",
                        "source_trajectory": str(trajectory_path),
                    }
                    if projected is not None
                    else {}
                ),
            },
        )


class BrowserUseExecutor(Executor):
    """Optional Browser Use adapter derived from the original benchmark runner."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "browser-use",
        browser: str = "local-headless",
        use_vision: bool = True,
        use_thinking: bool = True,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4096,
        add_schema_to_system_prompt: bool = False,
        dont_force_structured_output: bool = False,
        api_key_env: str = "OPENAI_API_KEY",
        openrouter_provider: str | None = None,
        provider_quantization: str | None = None,
        provider_sort: str | None = None,
        provider_require_parameters: bool = False,
        timeout_seconds: int = 1800,
    ):
        self.model = model
        self.provider = provider
        self.browser = browser
        self.use_vision = use_vision
        self.use_thinking = use_thinking
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.add_schema_to_system_prompt = add_schema_to_system_prompt
        self.dont_force_structured_output = dont_force_structured_output
        self.api_key_env = api_key_env
        self.openrouter_provider = openrouter_provider
        self.provider_quantization = provider_quantization
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.timeout_seconds = timeout_seconds

    def _llm(self):
        try:
            from browser_use import ChatGoogle
            from browser_use.llm import ChatAnthropic, ChatBrowserUse, ChatOpenAI
        except ImportError as exc:
            raise ExecutorError(
                "Browser Use is not installed; run `pip install -e '.[browser-use]'`"
            ) from exc
        if self.provider == "browser-use":
            return ChatBrowserUse(model=self.model)
        if self.provider == "openai":
            options: dict[str, Any] = {
                "model": self.model,
                "api_key": os.getenv(self.api_key_env),
                "base_url": self.base_url,
                "max_completion_tokens": self.max_output_tokens,
                "add_schema_to_system_prompt": self.add_schema_to_system_prompt,
                "dont_force_structured_output": self.dont_force_structured_output,
            }
            if self.reasoning_effort is not None:
                options["reasoning_effort"] = self.reasoning_effort
                options["reasoning_models"] = [self.model]
            llm = ChatOpenAI(**options)
            provider_options: dict[str, Any] = {
                "require_parameters": self.provider_require_parameters
            }
            if self.openrouter_provider:
                provider_options["only"] = [self.openrouter_provider]
                provider_options["allow_fallbacks"] = False
            if self.provider_quantization:
                provider_options["quantizations"] = [self.provider_quantization]
            if self.provider_sort:
                provider_options["sort"] = self.provider_sort
            if (
                self.openrouter_provider
                or self.provider_quantization
                or self.provider_sort
            ):
                original_get_client = llm.get_client

                def routed_client():
                    return _OpenAIClientProxy(
                        original_get_client(), {"provider": provider_options}
                    )

                llm.get_client = routed_client
            return llm
        if self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model, api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        if self.provider == "google":
            return ChatGoogle(model=self.model, api_key=os.getenv("GOOGLE_API_KEY"))
        raise ExecutorError(f"Unsupported model provider: {self.provider}")

    async def execute(
        self, task: TaskSpec, attempt: AttemptDescriptor
    ) -> ExecutionResult:
        try:
            from browser_use import Agent, Browser
        except ImportError as exc:
            raise ExecutorError(
                "Browser Use is not installed; run `pip install -e '.[browser-use]'`"
            ) from exc
        cdp_url = attempt.session.get("cdp_url")
        storage_state = attempt.session.get("storage_state")
        user_data_dir = attempt.session.get("user_data_dir")
        cloned_user_data_dir: Path | None = None
        if storage_state and self.browser == "browser-use-cloud":
            raise ExecutorError(
                "Prepared local storage_state cannot be sent to Browser Use Cloud; "
                "use local-headless/local-headful or provide a trusted CDP session"
            )
        if cdp_url:
            browser = Browser(
                cdp_url=str(cdp_url), downloads_path=attempt.artifact_dir
            )
        elif self.browser == "browser-use-cloud":
            browser = Browser(
                use_cloud=True,
                cloud_timeout=30,
                downloads_path=attempt.artifact_dir,
            )
        else:
            options: dict[str, Any] = {
                "headless": self.browser != "local-headful",
                "downloads_path": attempt.artifact_dir,
            }
            if storage_state:
                options["storage_state"] = Path(str(storage_state))
                options["user_data_dir"] = None
            if user_data_dir:
                cloned_user_data_dir = _clone_chrome_profile(
                    Path(str(user_data_dir)), prefix="rbbench-bu-chrome-"
                )
                options["user_data_dir"] = cloned_user_data_dir
            browser = Browser(**options)
        confirmed_task = task.confirmed_task.replace(
            "{{attempt_id}}", attempt.attempt_id
        )
        resolved_artifacts = dict(
            attempt.environment_data.get("resolved_fixture_artifacts", {})
        )
        available_file_paths = [
            str(Path(str(value)).resolve())
            for value in resolved_artifacts.values()
            if Path(str(value)).is_file()
        ]
        forbidden = "\n".join(f"- {item}" for item in task.safety.forbidden_actions)
        instruction = (
            f"Attempt ID: {attempt.attempt_id}\n"
            f"Start at {attempt.start_url}.\n\n{confirmed_task}\n\n"
            f"Fixture data: {json.dumps(task.fixture, ensure_ascii=False)}\n"
            "Resolved local input artifacts: "
            f"{json.dumps(resolved_artifacts)}\n\n"
            f"Forbidden actions:\n{forbidden or '- none'}\n\n"
            "When finished, return a concise answer containing every requested fact "
            "or a precise description of the completed action. Do not fabricate "
            "success or information."
        )
        started = time.perf_counter()
        try:
            await _start_browser_at(browser, attempt.start_url)
            llm = self._llm()
            raw_usage = (
                _install_openai_usage_tracker(llm)
                if self.provider == "openai"
                else {"reasoning_tokens": 0, "responses": 0}
            )
            agent = Agent(
                task=instruction,
                llm=llm,
                browser=browser,
                use_vision=self.use_vision,
                use_thinking=self.use_thinking,
                available_file_paths=available_file_paths,
            )
            history = await asyncio.wait_for(
                agent.run(), timeout=self.timeout_seconds
            )
            screenshot_dir = attempt.attempt_dir / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshots: list[str] = []
            for index, source in enumerate(history.screenshot_paths()):
                if source is None:
                    continue
                source_path = Path(source)
                if not source_path.exists():
                    continue
                destination = screenshot_dir / (
                    f"{index:04d}{source_path.suffix.lower() or '.png'}"
                )
                if source_path.resolve() != destination.resolve():
                    shutil.copy2(source_path, destination)
                screenshots.append(str(destination))
            usage = getattr(history, "usage", None)
            metrics: dict[str, Any] = {
                "steps": history.number_of_steps(),
                "duration_seconds": history.total_duration_seconds(),
                "cost": getattr(usage, "total_cost", 0.0) if usage else 0.0,
                "input_tokens": (
                    getattr(usage, "total_prompt_tokens", 0) if usage else 0
                ),
                "cached_input_tokens": (
                    getattr(usage, "total_prompt_cached_tokens", 0)
                    if usage
                    else 0
                ),
                "output_tokens": (
                    getattr(usage, "total_completion_tokens", 0) if usage else 0
                ),
                "reasoning_tokens": raw_usage["reasoning_tokens"],
                "non_reasoning_output_tokens": max(
                    0,
                    (getattr(usage, "total_completion_tokens", 0) if usage else 0)
                    - raw_usage["reasoning_tokens"],
                ),
                "total_tokens": (
                    getattr(usage, "total_tokens", 0) if usage else 0
                ),
                "model_invocations": (
                    getattr(usage, "entry_count", raw_usage["responses"])
                    if usage
                    else raw_usage["responses"]
                ),
            }
            final_result = history.final_result() or ""
            parsed = _json_object_from_text(final_result)
            observation = parsed or {
                "result": {"final_result": final_result},
                "safety": {},
            }
            observation.setdefault("safety", {})
            return ExecutionResult(
                final_result=final_result
                or "Agent did not return a final result",
                steps=[str(step) for step in history.agent_steps()],
                screenshots=screenshots,
                observation={**observation, "page": {"url": attempt.start_url}},
                metrics=metrics,
            )
        finally:
            try:
                await browser.stop()
            except Exception:
                pass
            if cloned_user_data_dir is not None:
                shutil.rmtree(cloned_user_data_dir, ignore_errors=True)
            elapsed = time.perf_counter() - started
            write_json(attempt.attempt_dir / "executor-timing.json", {"elapsed": elapsed})
