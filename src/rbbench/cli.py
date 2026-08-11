from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from .catalog import DEFAULT_CATALOG, REPO_ROOT, load_catalog
from .doctor import inspect_tasks
from .errors import BenchmarkError
from .executors import (
    BrowserAgentExecutor,
    BrowserUseExecutor,
    CommandExecutor,
    DryRunExecutor,
    Executor,
)
from .io import read_json, write_json
from .judges import CommandJudge, Judge, NativeLLMJudge
from .local_browser_agent import LocalBrowserAgentExecutor
from .runner import BenchmarkRunner


DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
DEFAULT_JUDGE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_REASONING_EFFORT = "high"
DEFAULT_JUDGE_OPENROUTER_PROVIDER = None

BENCHMARK_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _write_login_status(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _validate_benchmark_name(name: str) -> str:
    if not BENCHMARK_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "--name must start with a letter or digit and contain only letters, "
            "digits, '.', '_', or '-'"
        )
    return name


def _existing_benchmark_paths(
    name: str,
    *,
    results_dir: Path,
    runtime_dir: Path,
    task_ids: list[str],
) -> list[Path]:
    paths: list[Path] = []
    result_path = results_dir / name
    if result_path.exists():
        paths.append(result_path)
    for task_id in task_ids:
        attempt_path = runtime_dir / f"{name}-{task_id.lower()}"
        if attempt_path.exists():
            paths.append(attempt_path)
    return paths


def _confirm_benchmark_overwrite(name: str, paths: list[Path]) -> str:
    if not paths:
        return "overwrite"
    try:
        answer = input(
            f"Benchmark '{name}' already exists. Choose how to rerun it "
            "[overwrite/resume]: "
        )
    except EOFError as exc:
        raise ValueError(
            f"Benchmark '{name}' already exists; overwrite or resume was not selected"
        ) from exc
    action = answer.strip().lower()
    if action == "resume":
        return action
    if action != "overwrite":
        raise ValueError(
            f"Benchmark '{name}' already exists; choose 'overwrite' or 'resume'"
        )
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return action


def _judge_api_key_env(args: argparse.Namespace) -> str | None:
    configured = getattr(args, "judge_api_key_env", None)
    if configured:
        return str(configured)
    base_url = str(getattr(args, "judge_base_url", "") or "")
    hostname = (urlparse(base_url).hostname or "").lower()
    if args.judge_provider == "openai" and hostname == "openrouter.ai":
        return "OPENROUTER_API_KEY"
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(args.judge_provider)


def _task_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _browser_profile(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"--browser-profile must be an existing Chrome user-data directory: {path}"
        )
    return resolved


def _selected(args: argparse.Namespace):
    catalog = load_catalog(args.catalog)
    tasks = catalog.select(_task_ids(getattr(args, "task", None)))
    adapter = getattr(args, "adapter", None)
    category = getattr(args, "category", None)
    if adapter:
        tasks = [task for task in tasks if task.environment.adapter == adapter]
    if category:
        tasks = [task for task in tasks if task.category == category]
    return catalog, tasks


def _executor(args: argparse.Namespace) -> Executor:
    if bool(args.agent_cli) != bool(args.agent_config):
        raise ValueError("--agent-cli and --agent-config must be provided together")
    if args.agent_cli and args.executor != "browser-agent":
        raise ValueError(
            "--agent-cli and --agent-config require --executor browser-agent"
        )
    if args.executor == "dry-run":
        return DryRunExecutor()
    if args.executor == "command":
        if not args.executor_command:
            raise ValueError("--executor-command is required for command executor")
        return CommandExecutor(
            args.executor_command, timeout_seconds=args.timeout_seconds
        )
    if args.executor == "browser-agent":
        if args.agent_cli:
            return LocalBrowserAgentExecutor(
                cli_path=args.agent_cli,
                config_path=args.agent_config,
                timeout_seconds=args.timeout_seconds,
            )
        api_key = (
            os.getenv(args.agent_api_key_env)
            if args.agent_api_key_env
            else None
        )
        return BrowserAgentExecutor(
            provider=args.provider,
            model=args.model,
            endpoint_url=args.agent_base_url,
            reasoning_effort=args.agent_reasoning_effort,
            api_key=api_key,
            openrouter_provider=args.agent_openrouter_provider,
            max_model_len=args.agent_max_model_len,
            reserve_output_tokens=args.agent_reserve_output_tokens,
            headless=args.headless,
            executable_path=args.agent_chromium_path,
            max_steps=args.agent_max_steps,
            retry_count=args.agent_retry_count,
            timeout_seconds=args.timeout_seconds,
        )
    return BrowserUseExecutor(
        model=args.model,
        provider=args.provider,
        browser=args.browser,
        use_vision=not args.agent_text_only,
        use_thinking=not args.agent_no_thinking,
        base_url=args.agent_base_url,
        reasoning_effort=args.agent_reasoning_effort,
        max_output_tokens=args.agent_max_output_tokens,
        add_schema_to_system_prompt=args.agent_add_schema_to_system_prompt,
        dont_force_structured_output=args.agent_dont_force_structured_output,
        api_key_env=args.agent_api_key_env or "OPENAI_API_KEY",
        openrouter_provider=args.agent_openrouter_provider,
        provider_quantization=args.agent_provider_quantization,
        provider_sort=args.agent_provider_sort,
        provider_require_parameters=args.agent_provider_require_parameters,
        timeout_seconds=args.timeout_seconds,
    )


def _judge(args: argparse.Namespace) -> Judge:
    if args.judge == "command":
        if not args.judge_command:
            raise ValueError("--judge-command is required for command judge")
        return CommandJudge(args.judge_command, reference_dir=args.reference_dir)
    api_key_env = _judge_api_key_env(args)
    api_key = os.getenv(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise ValueError(f"Set {api_key_env} for the native LLM judge")
    base_url = args.judge_base_url
    hostname = (urlparse(base_url or "").hostname or "").lower()
    openrouter_provider = (
        args.judge_openrouter_provider
        if args.judge_provider == "openai" and hostname == "openrouter.ai"
        else None
    )
    return NativeLLMJudge(
        reference_dir=args.reference_dir,
        model=args.judge_model,
        provider=args.judge_provider,
        max_images=args.judge_max_images,
        text_only=args.judge_text_only,
        base_url=args.judge_base_url,
        reasoning_effort=args.judge_reasoning_effort,
        max_evidence_chars=args.judge_max_evidence_chars,
        max_output_tokens=args.judge_max_output_tokens,
        api_key=api_key,
        request_extra_body=(
            {
                "provider": {
                    "only": [openrouter_provider],
                    "allow_fallbacks": False,
                }
            }
            if openrouter_provider
            else None
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rbbench", description="BrowseWebApp bench"
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("codex-login", help="sign in to Codex with ChatGPT OAuth")

    sub.add_parser("validate", help="validate the 100-task catalog")

    listing = sub.add_parser("list", help="list benchmark tasks")
    listing.add_argument("--task", action="append")
    listing.add_argument("--adapter")
    listing.add_argument("--category")
    listing.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="show one task as JSON")
    show.add_argument("task_id")

    doctor = sub.add_parser("doctor", help="check references and lifecycle hooks")
    doctor.add_argument("--task", action="append")
    doctor.add_argument("--adapter")
    doctor.add_argument("--category")
    doctor.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "references" / "tasks")
    doctor.add_argument(
        "--executor", choices=("browser-agent", "command", "browser-use"),
        default="browser-agent",
    )
    doctor.add_argument("--executor-command")
    doctor.add_argument("--judge", choices=("llm", "command"), default="llm")
    doctor.add_argument("--judge-command")
    doctor.add_argument(
        "--judge-provider",
        choices=("openai", "anthropic", "google"),
        default=DEFAULT_JUDGE_PROVIDER,
    )
    doctor.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    doctor.add_argument("--judge-api-key-env")

    run = sub.add_parser("run", help="run selected tasks")
    run.add_argument("--task", action="append")
    run.add_argument("--adapter")
    run.add_argument("--category")
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument(
        "--name",
        required=True,
        help="unique name used for this benchmark run and its stored data",
    )
    run.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / ".runs" / "attempts")
    run.add_argument("--results-dir", type=Path, default=REPO_ROOT / ".runs" / "results")
    run.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "references" / "tasks")
    run.add_argument(
        "--executor",
        choices=("browser-agent", "dry-run", "command", "browser-use"),
        default="browser-agent",
    )
    run.add_argument("--executor-command")
    run.add_argument("--model", default="nvidia/GLM-5.2-NVFP4")
    run.add_argument(
        "--provider",
        choices=(
            "vllm", "together", "openrouter", "openai", "codex",
            "anthropic", "google", "browser-use",
        ),
        default="vllm",
    )
    run.add_argument(
        "--browser", choices=("local-headless", "local-headful", "browser-use-cloud"), default="local-headless"
    )
    run.add_argument(
        "--agent-text-only",
        action="store_true",
        help="disable Browser Use screenshot inputs for text-only agent models",
    )
    run.add_argument(
        "--agent-no-thinking",
        action="store_true",
        help="disable Browser Use's explicit thinking field for simpler model output",
    )
    run.add_argument("--agent-base-url")
    run.add_argument(
        "--agent-cli",
        type=Path,
        help="run a local browser-agent src/cli.ts, dist/cli.js, or native CLI",
    )
    run.add_argument(
        "--agent-config",
        type=Path,
        help="ordinary browser-agent YAML used by --agent-cli",
    )
    run.add_argument(
        "--agent-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max", "enabled"),
        default="high",
    )
    run.add_argument("--agent-openrouter-provider")
    run.add_argument("--agent-max-model-len", type=int, default=48_000)
    run.add_argument("--agent-reserve-output-tokens", type=int, default=4_000)
    run.add_argument("--agent-max-steps", type=int, default=50)
    run.add_argument("--agent-retry-count", type=int, default=0)
    run.add_argument("--agent-chromium-path")
    run.add_argument(
        "--browser-profile",
        type=Path,
        help=(
            "Chrome user-data directory to seed into each attempt's browser "
            "(logged-in cookies/local storage). Chrome must not be using the "
            "directory concurrently."
        ),
    )
    run.add_argument("--headless", action="store_true")
    run.add_argument("--agent-max-output-tokens", type=int, default=4096)
    run.add_argument("--agent-add-schema-to-system-prompt", action="store_true")
    run.add_argument("--agent-dont-force-structured-output", action="store_true")
    run.add_argument("--agent-api-key-env")
    run.add_argument(
        "--agent-provider-quantization",
        choices=("int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32"),
    )
    run.add_argument(
        "--agent-provider-sort", choices=("price", "throughput", "latency")
    )
    run.add_argument("--agent-provider-require-parameters", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument(
        "--judge",
        choices=("llm", "command"),
        default="llm",
        help="native LLM judge or external judge command",
    )
    run.add_argument("--judge-command")
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument(
        "--judge-provider",
        choices=("openai", "anthropic", "google"),
        default=DEFAULT_JUDGE_PROVIDER,
    )
    run.add_argument("--judge-max-images", type=int, default=10)
    judge_evidence = run.add_mutually_exclusive_group()
    judge_evidence.add_argument(
        "--judge-text-only",
        dest="judge_text_only",
        action="store_true",
        help="omit screenshots from judge evidence (default)",
    )
    judge_evidence.add_argument(
        "--judge-with-images",
        dest="judge_text_only",
        action="store_false",
        help="attach recent screenshots to judge evidence",
    )
    run.set_defaults(judge_text_only=True)
    run.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    run.add_argument("--judge-api-key-env")
    run.add_argument(
        "--judge-openrouter-provider",
        default=DEFAULT_JUDGE_OPENROUTER_PROVIDER,
        help="pin OpenRouter judging to one provider and disable fallbacks",
    )
    run.add_argument(
        "--judge-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=DEFAULT_JUDGE_REASONING_EFFORT,
    )
    run.add_argument("--judge-max-evidence-chars", type=int, default=39_500)
    run.add_argument("--judge-max-output-tokens", type=int, default=4_000)

    capture = sub.add_parser(
        "capture-reference", help="install a reviewed canonical observation"
    )
    capture.add_argument("task_id")
    capture.add_argument("observation", type=Path)
    capture.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "references" / "tasks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "codex-login":
            try:
                from browser_agent import ensure_codex_login
            except ImportError as exc:
                raise ValueError(
                    "Browser Agent SDK with Codex login support is not installed; "
                    "upgrade `browser-agent-python-sdk`"
                ) from exc
            try:
                asyncio.run(
                    ensure_codex_login(
                        on_log=lambda entry: _write_login_status(entry.message)
                    )
                )
            except Exception as exc:
                raise BenchmarkError(f"Codex login failed: {exc}") from exc
            return 0
        if args.command == "validate":
            catalog = load_catalog(args.catalog)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": catalog.name,
                        "version": catalog.version,
                        "tasks": len(catalog.tasks),
                        "adapters": dict(catalog.adapter_counts()),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "show":
            catalog = load_catalog(args.catalog)
            print(json.dumps(catalog.by_id(args.task_id).to_dict(), indent=2))
            return 0
        if args.command == "list":
            _, tasks = _selected(args)
            if args.json:
                print(json.dumps([task.to_dict() for task in tasks], indent=2))
            else:
                for task in tasks:
                    print(
                        f"{task.task_id}\t{task.environment.adapter}\t"
                        f"{task.category}\t{task.title}"
                    )
            return 0
        if args.command == "doctor":
            _, tasks = _selected(args)
            report = inspect_tasks(
                tasks,
                reference_dir=args.reference_dir,
                executor=args.executor,
                executor_command=args.executor_command,
                judge=args.judge,
                judge_command=args.judge_command,
                judge_provider=args.judge_provider,
                judge_api_key_env=_judge_api_key_env(args),
            )
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 2
        if args.command == "capture-reference":
            catalog = load_catalog(args.catalog)
            task = catalog.by_id(args.task_id)
            if task.oracle.reference_key != args.task_id:
                raise ValueError(f"{args.task_id} does not use a same-id reference")
            payload = read_json(args.observation)
            write_json(args.reference_dir / f"{args.task_id}.json", payload)
            print(args.reference_dir / f"{args.task_id}.json")
            return 0
        if args.command == "run":
            catalog, tasks = _selected(args)
            benchmark_name = _validate_benchmark_name(args.name)
            if not tasks:
                raise ValueError("No tasks matched the selection")
            existing_paths = _existing_benchmark_paths(
                benchmark_name,
                results_dir=args.results_dir,
                runtime_dir=args.runtime_dir,
                task_ids=[task.task_id for task in catalog.tasks],
            )
            existing_action = _confirm_benchmark_overwrite(
                benchmark_name, existing_paths
            )
            runner = BenchmarkRunner(
                catalog=catalog,
                executor=_executor(args),
                judge=_judge(args),
                runtime_dir=args.runtime_dir,
                results_dir=args.results_dir,
                browser_profile=_browser_profile(args.browser_profile),
            )
            summary = asyncio.run(
                runner.run(
                    tasks,
                    parallel=args.parallel,
                    run_id=benchmark_name,
                    resume=existing_action == "resume",
                )
            )
            print(json.dumps(summary, indent=2))
            return 0
    except (BenchmarkError, KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    sys.exit(main())
