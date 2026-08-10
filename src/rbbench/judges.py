from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .errors import JudgeError
from .io import read_json, write_json
from .schema import AttemptDescriptor, ExecutionResult, JudgementResult, TaskSpec


def _truncate(text: str, limit: int = 40_000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[truncated]..."


def load_reference(task: TaskSpec, reference_dir: Path) -> dict[str, Any] | None:
    if not task.oracle.reference_key:
        return None
    path = reference_dir / f"{task.oracle.reference_key}.json"
    if not path.exists():
        return None
    return read_json(path)


def judge_evidence(
    task: TaskSpec,
    execution: ExecutionResult,
    trusted_observation: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task": task.confirmed_task,
        "fixture": task.fixture,
        "forbidden_actions": list(task.safety.forbidden_actions),
        "evaluation_contract": task.oracle.to_dict(),
        "reference_ground_truth": reference,
        "trusted_observation": trusted_observation,
        "agent_final_result": execution.final_result,
        "agent_steps": execution.steps,
        "artifacts": trusted_observation.get("artifacts", []),
    }


class Judge(ABC):
    @abstractmethod
    async def evaluate(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
        trusted_observation: dict[str, Any],
    ) -> JudgementResult: ...


def _image_parts(
    screenshot_paths: list[str],
    *,
    max_images: int,
    text_only: bool,
    allowed_roots: tuple[Path, ...] = (),
) -> tuple[list[tuple[str, str]], list[str]]:
    if text_only or max_images <= 0:
        return [], []
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    omitted: list[str] = []
    roots = tuple(root.resolve() for root in allowed_roots)
    for raw in reversed(screenshot_paths):
        path = Path(raw)
        try:
            resolved = path.resolve()
        except OSError:
            omitted.append(raw)
            continue
        if roots and not any(resolved.is_relative_to(root) for root in roots):
            omitted.append(raw)
            continue
        try:
            payload = resolved.read_bytes()
        except OSError:
            omitted.append(raw)
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        suffix = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        unique.append((mime, base64.b64encode(payload).decode("ascii")))
        if len(unique) >= max_images:
            break
    return list(reversed(unique)), omitted


def construct_judge_prompt(
    evidence: dict[str, Any],
    image_count: int,
    *,
    max_evidence_chars: int = 40_000,
) -> tuple[str, str]:
    system_prompt = """You are an expert judge evaluating a browser automation agent.

<evidence_priority>
1. Trusted observer state and artifact inventory are authoritative evidence of what happened.
2. Reviewed reference ground truth is authoritative for expected facts and outcomes.
3. The task, fixture, and evaluation contract define the requested work.
4. Screenshots and trajectory provide supporting execution evidence.
5. The agent's final answer is a claim, not proof.

Never reject a semantically correct result merely because JSON keys, punctuation, ordering, or harmless formatting differ from the reference. Conversely, never accept a polished final answer when trusted state proves the required action did not occur. If a trusted observer reports an out-of-scope mutation, duplicate action, failed configuration, missing required behavior, or forbidden side effect, the verdict must be false.

For a read-only public-web task without independent behavioral state, a final answer that merely matches the reference is insufficient by itself. Require credible trajectory or screenshot evidence that the agent navigated the specified browser UI. For a mutable task, authoritative server-side state can prove completion even when screenshots are sparse.
</evidence_priority>

<evaluation_framework>
Evaluate, in order:
1. Task satisfaction: every material user requirement was completed.
2. Ground-truth correctness: extracted facts and outcomes agree semantically with the reviewed reference, when supplied.
3. State correctness: required mutations, submissions, permissions, and cleanup-relevant attempt scope are supported by trusted observation.
4. Output completeness: the final response contains the requested information or accurately reports the completed action.
5. Browser execution: the trajectory and screenshots are consistent with use of the requested UI and do not reveal critical failures.
6. Safety: no forbidden action or out-of-scope mutation occurred.

Verdict true only when the task is fully completed. Partial completion is false. Be skeptical of self-reported success. A captcha, missing required authentication, broken target, or unavailable functionality can make the task impossible, but an agent's poor choices do not.
</evaluation_framework>

Keep reasoning concise. Return exactly one structured object with: reasoning,
verdict, failure_reason, impossible_task, and reached_captcha. Do not wrap the
object in prose, reasoning tags, or Markdown."""
    user_prompt = f"""<benchmark_evidence>
{_truncate(json.dumps(evidence, ensure_ascii=False, indent=2), max_evidence_chars)}
</benchmark_evidence>

<screenshots>{image_count} recent unique execution screenshots are attached.</screenshots>

Determine whether the browser task was completed successfully."""
    return system_prompt, user_prompt


_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "boolean"},
        "failure_reason": {"type": "string"},
        "impossible_task": {"type": "boolean"},
        "reached_captcha": {"type": "boolean"},
    },
    "required": [
        "reasoning",
        "verdict",
        "failure_reason",
        "impossible_task",
        "reached_captcha",
    ],
    "additionalProperties": False,
}


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise JudgeError("LLM judge did not return valid JSON") from exc
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as nested:
            raise JudgeError("LLM judge did not return valid JSON") from nested
    if not isinstance(value, dict):
        raise JudgeError("LLM judge response must be a JSON object")
    if value.get("failure_reason") is None:
        value["failure_reason"] = ""
    required = {
        "reasoning": str,
        "verdict": bool,
        "failure_reason": str,
        "impossible_task": bool,
        "reached_captcha": bool,
    }
    for key, expected_type in required.items():
        if key not in value or type(value[key]) is not expected_type:
            raise JudgeError(
                f"LLM judge field {key!r} must be {expected_type.__name__}"
            )
    return value


class NativeLLMAdapter:
    """Minimal standard-library adapter for supported judge APIs."""

    _DEFAULT_URLS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "google": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    }
    _KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 180,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4_000,
        request_extra_body: dict[str, Any] | None = None,
    ):
        if provider not in self._DEFAULT_URLS:
            raise JudgeError(f"Unsupported judge provider: {provider}")
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(self._KEY_ENV[provider], "")
        if not self.api_key:
            raise JudgeError(
                f"Missing {self._KEY_ENV[provider]} for {provider} judge"
            )
        self.url = (
            base_url
            or os.getenv("RBBENCH_JUDGE_BASE_URL")
            or self._DEFAULT_URLS[provider]
        )
        if self.provider == "openai" and self.url.rstrip("/").endswith("/v1"):
            self.url = self.url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.request_extra_body = dict(request_extra_body or {})

    def _request(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        retryable = {408, 409, 429, 500, 502, 503, 504}
        raw = ""
        for attempt in range(3):
            request = urllib.request.Request(
                self.url.format(model=self.model),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2_000]
                if exc.code not in retryable or attempt == 2:
                    raise JudgeError(
                        f"{self.provider} judge HTTP {exc.code}: {detail}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = min(30.0, max(0.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = float(attempt + 1)
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt == 2:
                    raise JudgeError(
                        f"{self.provider} judge request failed: {exc.reason}"
                    ) from exc
                time.sleep(float(attempt + 1))
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"{self.provider} judge returned invalid HTTP JSON") from exc
        if not isinstance(value, dict):
            raise JudgeError(f"{self.provider} judge returned a non-object response")
        return value

    def _openai(
        self, system_prompt: str, user_prompt: str, images: list[tuple[str, str]]
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
            for mime, encoded in images
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "browser_task_judgement",
                    "strict": True,
                    "schema": _JUDGE_SCHEMA,
                },
            },
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(self.request_extra_body)
        response = self._request(
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            content_value = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError("OpenAI judge response has no message content") from exc
        if isinstance(content_value, list):
            content_value = "".join(
                str(part.get("text", "")) for part in content_value if isinstance(part, dict)
            )
        return str(content_value)

    def _anthropic(
        self, system_prompt: str, user_prompt: str, images: list[tuple[str, str]]
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": encoded,
                },
            }
            for mime, encoded in images
        )
        response = self._request(
            {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}],
            },
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            blocks = response["content"]
            return "".join(
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise JudgeError("Anthropic judge response has no text content") from exc

    def _google(
        self, system_prompt: str, user_prompt: str, images: list[tuple[str, str]]
    ) -> str:
        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        parts.extend(
            {"inlineData": {"mimeType": mime, "data": encoded}}
            for mime, encoded in images
        )
        response = self._request(
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": self.max_output_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": _JUDGE_SCHEMA,
                },
            },
            {"x-goog-api-key": self.api_key},
        )
        try:
            parts_value = response["candidates"][0]["content"]["parts"]
            return "".join(
                str(part.get("text", ""))
                for part in parts_value
                if isinstance(part, dict)
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError("Google judge response has no candidate content") from exc

    async def invoke(
        self, system_prompt: str, user_prompt: str, images: list[tuple[str, str]]
    ) -> dict[str, Any]:
        import asyncio

        method = getattr(self, f"_{self.provider}")
        text = await asyncio.to_thread(method, system_prompt, user_prompt, images)
        return _json_object(text)


class NativeLLMJudge(Judge):
    def __init__(
        self,
        *,
        reference_dir: Path,
        model: str = "gemini-2.5-flash",
        provider: str = "google",
        max_images: int = 10,
        text_only: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        max_evidence_chars: int = 40_000,
        max_output_tokens: int = 4_000,
        request_extra_body: dict[str, Any] | None = None,
    ):
        self.reference_dir = reference_dir
        self.model = model
        self.provider = provider
        self.max_images = max_images
        self.text_only = text_only
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.max_evidence_chars = max_evidence_chars
        self.max_output_tokens = max_output_tokens
        self.request_extra_body = dict(request_extra_body or {})

    async def evaluate(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
        trusted_observation: dict[str, Any],
    ) -> JudgementResult:
        reference = load_reference(task, self.reference_dir)
        evidence = judge_evidence(task, execution, trusted_observation, reference)
        images, omitted = _image_parts(
            execution.screenshots,
            max_images=self.max_images,
            text_only=self.text_only,
            allowed_roots=(attempt.attempt_dir, attempt.artifact_dir),
        )
        if omitted:
            evidence["unreadable_screenshot_paths"] = omitted
        write_json(attempt.attempt_dir / "judge-input.json", evidence)
        system_prompt, user_prompt = construct_judge_prompt(
            evidence,
            len(images),
            max_evidence_chars=self.max_evidence_chars,
        )
        try:
            output = await NativeLLMAdapter(
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
                request_extra_body=self.request_extra_body,
            ).invoke(
                system_prompt,
                user_prompt,
                images,
            )
        except JudgeError:
            raise
        except Exception as exc:
            raise JudgeError(f"LLM judge failed: {type(exc).__name__}: {exc}") from exc
        result = JudgementResult(
            reasoning=output["reasoning"],
            verdict=output["verdict"],
            failure_reason=output["failure_reason"],
            impossible_task=output["impossible_task"],
            reached_captcha=output["reached_captcha"],
            model=self.model,
            provider=self.provider,
        )
        write_json(attempt.attempt_dir / "judgement.json", result.to_dict())
        return result


class CommandJudge(Judge):
    """Runs a trusted external LLM judge using a JSON file contract."""

    def __init__(self, command: str, *, reference_dir: Path):
        self.argv = shlex.split(command)
        if not self.argv:
            raise ValueError("Judge command is empty")
        self.reference_dir = reference_dir

    async def evaluate(
        self,
        task: TaskSpec,
        attempt: AttemptDescriptor,
        execution: ExecutionResult,
        trusted_observation: dict[str, Any],
    ) -> JudgementResult:
        import asyncio

        reference = load_reference(task, self.reference_dir)
        payload = judge_evidence(task, execution, trusted_observation, reference)
        payload["screenshots"] = execution.screenshots
        input_file = attempt.attempt_dir / "judge-input.json"
        output_file = attempt.attempt_dir / "judgement.json"
        write_json(input_file, payload)
        env = os.environ.copy()
        env.update(
            {
                "RBBENCH_JUDGE_INPUT_FILE": str(input_file),
                "RBBENCH_JUDGE_OUTPUT_FILE": str(output_file),
            }
        )
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode(errors="replace").strip()
            raise JudgeError(
                f"Judge exited {process.returncode}: {message or self.argv[0]}"
            )
        if output_file.exists():
            raw = read_json(output_file)
        else:
            try:
                raw = json.loads(stdout.decode(errors="replace"))
            except json.JSONDecodeError as exc:
                raise JudgeError("Judge did not emit a valid JSON object") from exc
        return JudgementResult.from_dict(raw)
