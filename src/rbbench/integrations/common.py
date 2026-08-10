from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rbbench.catalog import REPO_ROOT
from rbbench.errors import InvalidEnvironmentError
from rbbench.io import read_json, write_json


TOKEN = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def context() -> dict[str, Any]:
    path = os.getenv("RBBENCH_CONTEXT_FILE")
    if not path:
        raise InvalidEnvironmentError("RBBENCH_CONTEXT_FILE is not set")
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise InvalidEnvironmentError("Hook context must be a JSON object")
    return payload


def emit(payload: dict[str, Any]) -> None:
    path = os.getenv("RBBENCH_OUTPUT_FILE")
    if path:
        write_json(Path(path), payload)
    else:
        print(json.dumps(payload))


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise InvalidEnvironmentError(f"Required environment variable is not set: {name}")
    return value


def configured_path(name: str, default: Path | None = None) -> Path:
    raw = os.getenv(name)
    if raw:
        path = Path(raw).expanduser()
    elif default is not None:
        path = default
    else:
        raise InvalidEnvironmentError(f"Required path is not set: {name}")
    return path.resolve()


def render(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return TOKEN.sub(lambda match: str(variables.get(match.group(1), match.group(0))), value)
    if isinstance(value, list):
        return [render(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(render(item, variables) for item in value)
    if isinstance(value, dict):
        return {key: render(item, variables) for key, item in value.items()}
    return value


def attempt_variables(payload: dict[str, Any]) -> dict[str, Any]:
    attempt = payload.get("attempt", {})
    return {
        "attempt_id": attempt.get("attempt_id", ""),
        "task_id": attempt.get("task_id", ""),
    }


class ExternalCommandError(InvalidEnvironmentError):
    pass


def run_command(
    argv: Iterable[str | Path],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int = 900,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in argv]
    if not command:
        raise ExternalCommandError("External command is empty")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ExternalCommandError(f"Command is not installed: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ExternalCommandError(
            f"Command timed out after {timeout}s: {Path(command[0]).name}"
        ) from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        detail = detail[-2000:] if detail else "no diagnostic output"
        raise ExternalCommandError(
            f"{Path(command[0]).name} exited {completed.returncode}: {detail}"
        )
    return completed


def parse_json_output(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    text = completed.stdout.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ExternalCommandError("Command did not emit a JSON object") from exc
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested:
            raise ExternalCommandError("Command emitted malformed JSON") from nested
    if not isinstance(value, dict):
        raise ExternalCommandError("Command JSON output must be an object")
    return value


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: Any
    headers: dict[str, str]


class JsonHttpClient:
    """Small dependency-free JSON client for trusted setup and grading APIs."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Accept": "application/json", **(headers or {})}

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: Any = None,
        expected: tuple[int, ...] = (200,),
    ) -> HttpResponse:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        if query:
            encoded = urllib.parse.urlencode(query, doseq=True)
            url = f"{url}{'&' if '?' in url else '?'}{encoded}"
        data = None
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                status = response.status
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except urllib.error.URLError as exc:
            raise InvalidEnvironmentError(
                f"Unable to reach configured service at {urllib.parse.urlsplit(url).netloc}"
            ) from exc
        parsed: Any = None
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw.decode(errors="replace")
        if status not in expected:
            detail = parsed if isinstance(parsed, str) else json.dumps(parsed)
            raise InvalidEnvironmentError(
                f"{method} {urllib.parse.urlsplit(url).path} returned HTTP {status}: "
                f"{detail[:1000]}"
            )
        return HttpResponse(status=status, body=parsed, headers=response_headers)


def integration_fixture(*parts: str) -> Path:
    return REPO_ROOT.joinpath("fixtures", *parts)
