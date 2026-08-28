#!/usr/bin/env python3
"""RBBench command adapter for BrowserCode/Bcode."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any


BCODE_BIN = Path(os.getenv("BCODE_BIN", str(Path.home() / ".bcode/bin/bcode")))
CHROME_BIN = Path(
    os.getenv(
        "BCODE_CHROME_BIN",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
)
MODEL = os.getenv("BCODE_MODEL", "openrouter/z-ai/glm-5.2")
MODEL_VARIANT = os.getenv("BCODE_MODEL_VARIANT", "high")
OPENROUTER_PROVIDER = os.getenv("BCODE_OPENROUTER_PROVIDER")

_CHROME_PROFILE_TRANSIENT_NAMES = {
    "DevToolsActivePort",
    "SingletonCookie",
    "SingletonLock",
    "SingletonSocket",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def merge_config(existing: str | None, overlay: dict[str, Any]) -> str:
    base: dict[str, Any] = {}
    if existing:
        parsed = json.loads(existing)
        if isinstance(parsed, dict):
            base = parsed

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        result = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return json.dumps(merge(base, overlay), separators=(",", ":"))


def build_openrouter_provider_config() -> dict[str, Any]:
    if not MODEL.startswith("openrouter/") or not OPENROUTER_PROVIDER:
        return {}
    openrouter_model = MODEL.removeprefix("openrouter/")
    return {
        "provider": {
            "openrouter": {
                "models": {
                    openrouter_model: {
                        "options": {
                            "provider": {
                                "order": [OPENROUTER_PROVIDER],
                                "allow_fallbacks": False,
                            }
                        }
                    }
                }
            }
        }
    }


async def launch_chrome(
    profile_dir: Path, start_url: str = "about:blank"
) -> tuple[asyncio.subprocess.Process, str]:
    process = await asyncio.create_subprocess_exec(
        str(CHROME_BIN),
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
        "--remote-debugging-port=0",
        f"--user-data-dir={profile_dir}",
        start_url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    active_port = profile_dir / "DevToolsActivePort"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"Chrome exited before CDP was ready: {process.returncode}")
        if active_port.exists():
            lines = active_port.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2:
                return process, f"ws://127.0.0.1:{lines[0]}{lines[1]}"
        await asyncio.sleep(0.1)
    process.terminate()
    raise RuntimeError("Timed out waiting for Chrome DevToolsActivePort")


async def connect_browser(
    environment: dict[str, str],
    seed_profile: Path | None = None,
    start_url: str = "about:blank",
) -> tuple[asyncio.subprocess.Process | None, Path | None]:
    """Use an externally configured CDP browser or launch an isolated Chrome."""
    external_cdp = {
        name: value.strip()
        for name in ("BU_CDP_URL", "BU_CDP_WS")
        if (value := environment.get(name, "")).strip()
    }
    if len(external_cdp) > 1:
        raise ValueError("Set only one of BU_CDP_URL or BU_CDP_WS")
    if external_cdp:
        return None, None

    profile_dir = Path(tempfile.mkdtemp(prefix="rbbench-bcode-chrome-"))
    if seed_profile is not None:
        if not seed_profile.is_dir():
            raise ValueError(f"Chrome profile seed is not a directory: {seed_profile}")

        def ignore_transient(_directory: str, names: list[str]) -> set[str]:
            return _CHROME_PROFILE_TRANSIENT_NAMES.intersection(names)

        shutil.copytree(
            seed_profile,
            profile_dir,
            dirs_exist_ok=True,
            ignore=ignore_transient,
        )
    chrome, cdp_ws = await launch_chrome(profile_dir, start_url)
    environment["BU_CDP_WS"] = cdp_ws
    return chrome, profile_dir


async def stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def stop_chrome_profile(profile_dir: Path) -> None:
    """Stop Chrome descendants that macOS may re-parent after launcher exit."""
    marker = f"--user-data-dir={profile_dir}"
    for signal_name in ("TERM", "KILL"):
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/pkill",
            f"-{signal_name}",
            "-f",
            marker,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        if signal_name == "TERM":
            await asyncio.sleep(0.5)


def step_text(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    if event_type == "tool_use":
        tool = str(part.get("tool") or "tool")
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        inputs = state.get("input") if isinstance(state.get("input"), dict) else {}
        if tool in {"browser-execute", "browser_execute"}:
            return f"browser_execute: {str(inputs.get('code') or inputs.get('python') or '')[:4000]}"
        return f"{tool}: {json.dumps(inputs, ensure_ascii=False)[:4000]}"
    if event_type in {"text", "reasoning"}:
        text = str(part.get("text") or "").strip()
        return f"{event_type}: {text[:4000]}" if text else None
    if event_type == "error":
        return f"error: {json.dumps(event.get('error'), ensure_ascii=False)[:4000]}"
    return None


def usage_from_event(event: dict[str, Any]) -> dict[str, int | float]:
    if event.get("type") != "step_finish":
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "non_reasoning_output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else part
    uncached_input_tokens = int(
        tokens.get("input") or tokens.get("input_tokens") or 0
    )
    non_reasoning_output_tokens = int(
        tokens.get("output") or tokens.get("output_tokens") or 0
    )
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    cached_input_tokens = int(cache.get("read") or 0)
    input_tokens = uncached_input_tokens + cached_input_tokens
    raw_reasoning = tokens.get("reasoning") or 0
    reasoning_tokens = int(
        raw_reasoning if isinstance(raw_reasoning, (int, float)) else 0
    )
    output_tokens = non_reasoning_output_tokens + reasoning_tokens
    total = int(tokens.get("total") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "non_reasoning_output_tokens": non_reasoning_output_tokens,
        "total_tokens": total,
        "cost": float(part.get("cost") or 0.0),
    }


async def main() -> int:
    task = read_json(Path(os.environ["RBBENCH_TASK_FILE"]))
    attempt = read_json(Path(os.environ["RBBENCH_ATTEMPT_FILE"]))
    output_path = Path(os.environ["RBBENCH_OUTPUT_FILE"])
    attempt_dir = Path(attempt["attempt_dir"])
    artifact_dir = Path(attempt["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    events_path = artifact_dir / "bcode-events.jsonl"
    stderr_path = artifact_dir / "bcode-stderr.log"
    screenshot_dir = attempt_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(tempfile.mkdtemp(prefix="rbbench-bcode-workspace-"))

    confirmed = str(task["confirmed_task"]).replace(
        "{{attempt_id}}", str(attempt["attempt_id"])
    )
    forbidden = task.get("safety", {}).get("forbidden_actions", [])
    instruction = (
        "You are BrowserCode, operating a preconfigured browser autonomously. "
        "Use browser_execute and call await session.connect() with no arguments. "
        "Use only the rendered browser UI for the measured task; do not replace UI "
        "work with product APIs or direct HTTP requests. Take screenshots when useful.\n\n"
        f"Attempt ID: {attempt['attempt_id']}\n"
        f"Start URL: {attempt['start_url']}\n\n"
        f"{confirmed}\n\n"
        f"Task data: {json.dumps(task.get('fixture', {}), ensure_ascii=False)}\n"
        "Resolved local input artifacts: "
        f"{json.dumps(attempt.get('environment_data', {}).get('resolved_fixture_artifacts', {}), ensure_ascii=False)}\n"
        f"Download and output directory: {artifact_dir}\n"
        f"Forbidden actions: {json.dumps(forbidden, ensure_ascii=False)}\n\n"
        "When finished, return a concise answer containing every requested fact or a "
        "precise description of the completed action. Do not fabricate success."
    )

    chrome: asyncio.subprocess.Process | None = None
    profile_dir: Path | None = None
    bcode: asyncio.subprocess.Process | None = None
    started = time.monotonic()
    steps: list[str] = []
    final_text = ""
    errors: list[str] = []
    counts: Counter[str] = Counter()
    input_tokens = cached_input_tokens = output_tokens = 0
    reasoning_tokens = non_reasoning_output_tokens = total_tokens = 0
    cost = 0.0

    try:
        base_config = {"experimental": {"fetch_use": False}}
        config_content = merge_config(
            os.getenv("OPENCODE_CONFIG_CONTENT"), base_config
        )
        config_content = merge_config(
            config_content, build_openrouter_provider_config()
        )
        environment = {
            **os.environ,
            "BCODE_SCREENSHOT_DIR": str(screenshot_dir),
            "OPENCODE_CONFIG_CONTENT": config_content,
        }
        seed_profile_value = attempt.get("session", {}).get("user_data_dir")
        seed_profile = Path(str(seed_profile_value)) if seed_profile_value else None
        chrome, profile_dir = await connect_browser(
            environment, seed_profile, str(attempt["start_url"])
        )
        command = [
            str(BCODE_BIN),
            "run",
            "--model",
            MODEL,
            "--variant",
            MODEL_VARIANT,
            "--format",
            "json",
            "--dangerously-skip-permissions",
            "--dir",
            str(workspace_dir),
            "--",
            instruction,
        ]
        bcode = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace_dir,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=256 * 1024 * 1024,
        )
        assert bcode.stdout is not None
        assert bcode.stderr is not None

        async def drain_stderr() -> str:
            chunks: list[bytes] = []
            with stderr_path.open("wb", buffering=0) as stderr_log:
                while chunk := await bcode.stderr.read(64 * 1024):
                    stderr_log.write(chunk)
                    chunks.append(chunk)
                    if sum(map(len, chunks)) > 2 * 1024 * 1024:
                        chunks = [b"".join(chunks)[-2 * 1024 * 1024 :]]
            return b"".join(chunks).decode("utf-8", errors="replace")

        stderr_task = asyncio.create_task(drain_stderr())
        with events_path.open("wb", buffering=0) as events_log:
            while line := await bcode.stdout.readline():
                events_log.write(line)
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = str(event.get("type") or "unknown")
                counts[event_type] += 1
                if text := step_text(event):
                    steps.append(text)
                part = event.get("part") if isinstance(event.get("part"), dict) else {}
                if event_type == "text" and str(part.get("text") or "").strip():
                    final_text = str(part["text"]).strip()
                if event_type == "error":
                    errors.append(json.dumps(event.get("error"), ensure_ascii=False))
                values = usage_from_event(event)
                input_tokens += int(values["input_tokens"])
                cached_input_tokens += int(values["cached_input_tokens"])
                output_tokens += int(values["output_tokens"])
                reasoning_tokens += int(values["reasoning_tokens"])
                non_reasoning_output_tokens += int(
                    values["non_reasoning_output_tokens"]
                )
                total_tokens += int(values["total_tokens"])
                cost += float(values["cost"])
        await bcode.wait()
        stderr = await stderr_task
        if bcode.returncode != 0 and not final_text:
            raise RuntimeError(
                f"bcode exited {bcode.returncode}: {stderr[-2000:] or errors[-1:] }"
            )
    finally:
        await stop_process(bcode)
        await stop_process(chrome)
        if profile_dir is not None:
            await stop_chrome_profile(profile_dir)
            shutil.rmtree(profile_dir, ignore_errors=True)
        shutil.rmtree(workspace_dir, ignore_errors=True)

    screenshots = [
        str(path)
        for path in sorted(screenshot_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    result = {
        "final_result": final_text or (errors[-1] if errors else "[bcode_no_output]"),
        "steps": steps,
        "screenshots": screenshots,
        "observation": {
            "result": {"final_result": final_text},
            "page": {"url": attempt["start_url"]},
            "safety": {},
        },
        "metrics": {
            "steps": counts.get("step_finish", 0),
            "duration_seconds": time.monotonic() - started,
            "cost": cost,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "non_reasoning_output_tokens": non_reasoning_output_tokens,
            "total_tokens": total_tokens,
            "model_invocations": counts.get("step_finish", 0),
            "event_counts": dict(counts),
            "error_count": len(errors),
            "model": MODEL,
            "model_variant": MODEL_VARIANT,
        },
        "error": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
