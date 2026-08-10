"""Project browser-agent trajectories into compact independent-judge evidence.

It consumes pre-verdict success-verifier context but never copies the agent's
own verifier verdict into the independent judge input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_EVIDENCE_CHARS = 39_500
_TOP_LEVEL_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9]*):")
_CHILD_KEY = re.compile(r"^  ([A-Za-z][A-Za-z0-9]*):")
_HISTORY_ENTRY = re.compile(r"^  - role:")
_DIFF_HUNK = re.compile(
    r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@$"
)


@dataclass(frozen=True)
class BrowserAgentExecution:
    final_result: str
    steps: list[str]
    num_steps: int
    duration_seconds: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    non_reasoning_output_tokens: int
    total_tokens: int
    model_invocations: int


def _usage_int(value: Any) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    return max(0, int(value))


def _aggregate_model_usage(artifact: dict[str, Any]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "non_reasoning_output_tokens": 0,
        "total_tokens": 0,
        "model_invocations": 0,
    }
    invocations = artifact.get("modelInvocations")
    if not isinstance(invocations, list):
        return totals
    for invocation in invocations:
        if not isinstance(invocation, dict):
            continue
        usage = invocation.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens = _usage_int(usage.get("input_tokens"))
        cached_input_tokens = _usage_int(usage.get("cached_input_tokens"))
        reasoning_tokens = _usage_int(usage.get("reasoning_tokens"))
        output_tokens = _usage_int(usage.get("output_tokens"))
        non_reasoning_output_tokens = _usage_int(
            usage.get("non_reasoning_output_tokens")
        )
        if output_tokens == 0:
            output_tokens = reasoning_tokens + non_reasoning_output_tokens
        if non_reasoning_output_tokens == 0 and output_tokens >= reasoning_tokens:
            non_reasoning_output_tokens = output_tokens - reasoning_tokens
        total_tokens = _usage_int(usage.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        totals["input_tokens"] += input_tokens
        totals["cached_input_tokens"] += cached_input_tokens
        totals["output_tokens"] += output_tokens
        totals["reasoning_tokens"] += reasoning_tokens
        totals["non_reasoning_output_tokens"] += non_reasoning_output_tokens
        totals["total_tokens"] += total_tokens
        totals["model_invocations"] += 1
    return totals


def load_trajectory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing browser-agent trajectory: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path} at line {line_number}")
        records.append(record)
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one browser-agent trajectory in {path}; "
            f"found {len(records)}"
        )
    return records[0]


def load_token_usage(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing browser-agent token usage: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(artifact, dict) or artifact.get("schemaVersion") not in {1, 2}:
        raise ValueError(f"Invalid browser-agent token usage artifact: {path}")
    raw_totals = artifact.get("totals")
    attempts = artifact.get("attempts")
    if not isinstance(raw_totals, dict) or not isinstance(attempts, list):
        raise ValueError(f"Incomplete browser-agent token usage artifact: {path}")
    totals = {
        key: _usage_int(raw_totals.get(key))
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "non_reasoning_output_tokens",
            "total_tokens",
        )
    }
    totals["model_invocations"] = sum(
        1
        for attempt in attempts
        if isinstance(attempt, dict)
        for invocation in (
            attempt.get("invocations")
            if isinstance(attempt.get("invocations"), list)
            else []
        )
        if isinstance(invocation, dict) and isinstance(invocation.get("usage"), dict)
    )
    return totals


def _sections(text: str, pattern: re.Pattern[str], *, skip_first: bool = False):
    lines = text.splitlines(keepends=True)
    offset = 1 if skip_first else 0
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines[offset:], start=offset):
        match = pattern.match(line)
        if match:
            starts.append((match.group(1), index))
    result: list[tuple[str, str]] = []
    for position, (key, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        result.append((key, "".join(lines[start:end]).rstrip()))
    return result


def _top_level_sections(text: str) -> dict[str, str]:
    return dict(_sections(text, _TOP_LEVEL_KEY))


def _child_sections(text: str) -> list[tuple[str, str]]:
    return _sections(text, _CHILD_KEY, skip_first=True)


def _history_entries(section: str) -> list[str]:
    lines = section.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines[1:], start=1)
        if _HISTORY_ENTRY.match(line)
    ]
    return [
        "".join(lines[start : starts[position + 1] if position + 1 < len(starts) else len(lines)]).rstrip()
        for position, start in enumerate(starts)
    ]


def _truncate_middle(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    marker = f"\n...[{label} omitted: {len(text) - limit:,} chars]...\n"
    if len(marker) >= limit:
        return marker[:limit]
    remaining = limit - len(marker)
    head = (remaining * 2) // 3
    return text[:head] + marker + text[-(remaining - head) :]


def _validator_prompt(artifact: dict[str, Any]) -> str | None:
    invocations = artifact.get("modelInvocations")
    if not isinstance(invocations, list):
        return None
    for invocation in reversed(invocations):
        if not isinstance(invocation, dict) or invocation.get("stage") != "verifySuccess":
            continue
        messages = invocation.get("messages")
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return message["content"]
    return None


def _sanitized_final_step(section: str) -> str:
    children = _child_sections(section)
    if not children:
        return section
    kept: list[str] = []
    for key, child in children:
        if key in {"thinking", "previousStepPlanUpdate"}:
            continue
        if key == "result":
            kept.append("  result: '[same terminal result supplied separately]'")
        else:
            kept.append(child)
    return "finalStep:\n" + "\n".join(kept)


def _final_state(section: str) -> tuple[str, str]:
    metadata: list[str] = []
    page_state = ""
    for key, child in _child_sections(section):
        if key in {"html", "projection"}:
            page_state = child
        elif key not in {"task", "latestUserPromptTokenCount"}:
            metadata.append(child)
    return "finalPromptPayload:\n" + "\n".join(metadata), page_state


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    for part in content:
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ):
            return part["text"]
    return None


def _top_level_scalar(text: str, key: str) -> str | None:
    lines = text.splitlines()
    prefix = f"{key}:"
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        raw = line[len(prefix) :].strip()
        if raw in {"''", '""'}:
            return ""
        if raw not in {"|", "|-", ">", ">-"}:
            return raw.strip("\"'")
        block: list[str] = []
        for child in lines[index + 1 :]:
            if child and not child.startswith(" "):
                break
            block.append(child[2:] if child.startswith("  ") else "")
        if raw.startswith("|"):
            return "\n".join(block).rstrip("\n")

        # js-yaml emits folded scalars with blank lines between source lines
        # when it needs to preserve their newlines. Adjacent non-empty lines
        # are ordinary wrapped text and fold to a space.
        folded = ""
        previous_more_indented = False
        pending_blank_lines = 0
        for child in block:
            if child == "":
                pending_blank_lines += 1
                continue
            more_indented = child.startswith(" ")
            if folded:
                if pending_blank_lines:
                    folded += "\n" * pending_blank_lines
                elif previous_more_indented or more_indented:
                    folded += "\n"
                else:
                    folded += " "
            folded += child
            previous_more_indented = more_indented
            pending_blank_lines = 0
        return folded
    return None


def _apply_unified_projection_diff(base: str, delta: str) -> str:
    if not delta.strip():
        return base
    base_lines = base.splitlines()
    delta_lines = delta.splitlines()
    output: list[str] = []
    base_index = 0
    index = 2 if delta_lines[:2] == [
        "--- previous-projection",
        "+++ current-projection",
    ] else 0
    while index < len(delta_lines):
        match = _DIFF_HUNK.match(delta_lines[index])
        if not match:
            raise ValueError("Invalid cumulative semantic projection diff")
        old_start = int(match.group(1)) - 1
        output.extend(base_lines[base_index:old_start])
        base_index = old_start
        index += 1
        while index < len(delta_lines) and not delta_lines[index].startswith("@@ "):
            line = delta_lines[index]
            if not line:
                raise ValueError("Invalid empty line in semantic projection diff")
            marker, value = line[0], line[1:]
            if marker == " ":
                if base_index >= len(base_lines) or base_lines[base_index] != value:
                    raise ValueError("Semantic projection diff context does not match")
                output.append(value)
                base_index += 1
            elif marker == "-":
                if base_index >= len(base_lines) or base_lines[base_index] != value:
                    raise ValueError("Semantic projection diff deletion does not match")
                base_index += 1
            elif marker == "+":
                output.append(value)
            else:
                raise ValueError("Invalid semantic projection diff operation")
            index += 1
    output.extend(base_lines[base_index:])
    return "\n".join(output)


def _reconstructed_projection(artifact: dict[str, Any]) -> str | None:
    steps = artifact.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        return None
    messages = steps[-1].get("messages")
    if not isinstance(messages, list):
        return None
    projection: str | None = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = _message_text(message)
        if content is None:
            continue
        value = _top_level_scalar(content, "projection")
        if value is None:
            continue
        mode = _top_level_scalar(content, "projectionContextMode")
        if mode == "delta":
            if projection is None:
                continue
            projection = _apply_unified_projection_diff(projection, value)
        else:
            projection = value
    return projection


def _recent_history(entries: list[str], budget: int) -> str:
    if not entries or budget <= 0:
        return ""
    heading = "recentStepHistory:\n"
    selected: list[str] = []
    used = len(heading)
    for entry in reversed(entries):
        separator = 1 if selected else 0
        if used + separator + len(entry) <= budget:
            selected.append(entry)
            used += separator + len(entry)
        else:
            if not selected:
                selected.append(
                    _truncate_middle(entry, max(0, budget - used - 1), "history")
                )
            break
    selected.reverse()
    omitted = len(entries) - len(selected)
    note = f"  # Earlier history omitted: {omitted} entries\n" if omitted else ""
    return _truncate_middle(
        heading + note + "\n".join(selected), budget, "older step history"
    )


def _project_validator_prompt(
    prompt: str,
    max_chars: int,
    reconstructed_projection: str | None = None,
) -> str:
    sections = _top_level_sections(prompt)
    if "finalStep" not in sections or "finalPromptPayload" not in sections:
        return _truncate_middle(prompt, max_chars, "validator evidence")
    executed = sections.get("executedSteps", "executedSteps: unknown")
    final_step = _sanitized_final_step(sections["finalStep"])
    state, page_state = _final_state(sections["finalPromptPayload"])
    if reconstructed_projection is not None:
        page_state = "  projection: |-\n" + "\n".join(
            f"    {line}" for line in reconstructed_projection.splitlines()
        )
    history = _history_entries(sections.get("stepHistory", ""))
    header = (
        "browserAgentEvaluatorEvidence:\n"
        "  source: pre-verdict browser-agent validator context\n"
        "  note: final result is separate; browser-agent verdict is excluded"
    )
    history_reserve = min(8_000, max_chars // 4)
    primary_budget = max(0, max_chars - len(header) - len(executed) - 8 - history_reserve)
    if len(final_step) + len(state) > primary_budget:
        state_budget = min(len(state), primary_budget, max(4_000, primary_budget // 2))
        state = _truncate_middle(state, state_budget, "final prompt metadata")
        final_step = _truncate_middle(
            final_step,
            max(0, primary_budget - len(state) - 2),
            "terminal tool payload",
        )
    primary = "\n\n".join((header, executed, final_step, state))
    parts = [primary]
    history_block = _recent_history(
        history, min(history_reserve, max(0, max_chars - len(primary) - 2))
    )
    if history_block:
        parts.append(history_block)
    remaining = max_chars - len("\n\n".join(parts)) - 2
    if page_state and remaining > 200:
        parts.append(
            _truncate_middle(
                "finalPageState:\n" + page_state,
                remaining,
                "final page state",
            )
        )
    return _truncate_middle("\n\n".join(parts), max_chars, "evaluator evidence")


def _fallback_evidence(artifact: dict[str, Any], max_chars: int) -> str:
    steps = artifact.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        return "Browser-agent trajectory contains no usable final step."
    last = steps[-1]
    messages = last.get("messages")
    serializable = messages if isinstance(messages, list) else []
    text = json.dumps(serializable, ensure_ascii=False, indent=2)
    header = (
        "browserAgentEvaluatorEvidence:\n"
        "  source: final browser-agent step (no validator invocation)\n"
        f"  completed: {str(bool(artifact.get('completed'))).lower()}\n"
        f"  step: {last.get('step', 'unknown')}\n"
    )
    return header + _truncate_middle(text, max(0, max_chars - len(header)), "messages")


def convert_trajectory(
    artifact: dict[str, Any],
    *,
    expected_task: str,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    usage_totals: dict[str, int] | None = None,
) -> BrowserAgentExecution:
    if artifact.get("task") != expected_task:
        raise ValueError("Browser-agent trajectory task does not match executor task")
    prompt = _validator_prompt(artifact)
    reconstructed_projection = _reconstructed_projection(artifact)
    evidence = (
        _project_validator_prompt(
            prompt,
            max_evidence_chars,
            reconstructed_projection,
        )
        if prompt is not None
        else _fallback_evidence(artifact, max_evidence_chars)
    )
    final_result = artifact.get("finalResult")
    if not isinstance(final_result, str) or not final_result.strip():
        final_result = "Agent did not return a final result."
    raw_steps = artifact.get("steps")
    stored_count = len(raw_steps) if isinstance(raw_steps, list) else 0
    try:
        num_steps = int(artifact.get("browserEquivalentSteps", stored_count))
    except (TypeError, ValueError):
        num_steps = stored_count
    duration = artifact.get("trajectoryDurationMs", artifact.get("durationMs", 0))
    if not isinstance(duration, (int, float)):
        duration = 0
    usage = usage_totals or _aggregate_model_usage(artifact)
    return BrowserAgentExecution(
        final_result=final_result,
        steps=[evidence],
        num_steps=num_steps,
        duration_seconds=float(duration) / 1000,
        **usage,
    )
