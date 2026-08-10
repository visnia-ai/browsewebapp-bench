from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from rbbench.catalog import REPO_ROOT
from rbbench.errors import InvalidEnvironmentError
from rbbench.io import read_json

from .base import Integration
from .common import JsonHttpClient, render


DEFAULT_CONFIG = REPO_ROOT / "configs" / "tally" / "forms.json"
DEFAULT_TOKEN_FILE = REPO_ROOT / "session-pools" / "private" / "tally-api-token"


def _canonical(value: Any) -> Any:
    """Normalize Tally's typed answers without discarding file metadata."""

    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False)


class TallyIntegration(Integration):
    """Lifecycle adapter for public Tally forms with trusted API grading.

    Respondents never receive the private Tally account or API token. Every form
    contains hidden ``attempt_id`` and ``task_id`` fields populated by the start
    URL. Those values are the isolation boundary for observation and cleanup.
    """

    def __init__(self, client: JsonHttpClient | None = None):
        self._injected_client = client
        self._form_cache: dict[str, dict[str, Any]] = {}

    def _config_path(self) -> Path:
        raw = os.getenv("TALLY_FORMS_CONFIG")
        return Path(raw).expanduser().resolve() if raw else DEFAULT_CONFIG

    def _config(self) -> dict[str, Any]:
        path = self._config_path()
        if not path.exists():
            raise InvalidEnvironmentError(f"Tally form configuration is missing: {path}")
        payload = read_json(path)
        if not isinstance(payload.get("forms"), dict):
            raise InvalidEnvironmentError("Tally configuration must contain a forms object")
        return payload

    def _form(self, task_id: str) -> dict[str, Any]:
        form = self._config()["forms"].get(task_id)
        if not isinstance(form, dict):
            raise InvalidEnvironmentError(f"No provisioned Tally form for {task_id}")
        if not form.get("form_id") or str(form["form_id"]).startswith("REPLACE_"):
            raise InvalidEnvironmentError(f"Tally form id is not provisioned for {task_id}")
        return form

    def _client(self) -> JsonHttpClient:
        if self._injected_client is not None:
            return self._injected_client
        token = os.getenv("TALLY_API_TOKEN")
        token_file = Path(os.getenv("TALLY_API_TOKEN_FILE", DEFAULT_TOKEN_FILE))
        if not token and token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise InvalidEnvironmentError("Required environment variable is not set: TALLY_API_TOKEN")
        return JsonHttpClient(
            "https://api.tally.so",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "rbbench-lifecycle/0.1 (+https://github.com/visnia-ai/browser-agent)",
                "tally-version": "2026-06-23",
            },
        )

    @staticmethod
    def _question_labels(questions: list[dict[str, Any]]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for question in questions:
            title = str(question.get("title") or question.get("name") or "").strip()
            if not title:
                continue
            for key in ("id", "uuid", "blockGroupUuid"):
                if question.get(key):
                    labels[str(question[key])] = title
            for field in question.get("fields", []):
                if not isinstance(field, dict):
                    continue
                for key in ("id", "uuid", "blockGroupUuid"):
                    if field.get(key):
                        labels[str(field[key])] = title
        return labels

    @classmethod
    def _answers(
        cls,
        submission: dict[str, Any],
        questions: list[dict[str, Any]],
        option_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        labels = cls._question_labels(questions)
        metadata = {str(item.get("id")): item for item in questions if item.get("id")}
        option_labels = option_labels or {}

        def translate(value: Any) -> Any:
            if isinstance(value, str):
                return option_labels.get(value, value)
            if isinstance(value, list):
                return [translate(item) for item in value]
            if isinstance(value, dict):
                return {str(key): translate(item) for key, item in value.items()}
            return value

        answers: dict[str, Any] = {}
        for response in submission.get("responses", []):
            if not isinstance(response, dict):
                continue
            question_id = str(response.get("questionId") or response.get("question_id") or "")
            label = labels.get(question_id, question_id)
            value = response.get("formattedAnswer")
            if value is None:
                value = response.get("answer")
            value = translate(value)
            question = metadata.get(question_id, {})
            if question.get("type") == "HIDDEN_FIELDS" and isinstance(value, dict):
                answers.update(value)
                continue
            if (
                question.get("type") in {"DROPDOWN", "MULTIPLE_CHOICE"}
                and isinstance(value, list)
                and len(value) == 1
            ):
                value = value[0]
            if label in answers:
                previous = answers[label]
                answers[label] = previous + [value] if isinstance(previous, list) else [previous, value]
            else:
                answers[label] = _canonical(value)
        return answers

    def _form_metadata(self, form_id: str) -> dict[str, Any]:
        if form_id not in self._form_cache:
            response = self._client().request("GET", f"/forms/{form_id}")
            if not isinstance(response.body, dict):
                raise InvalidEnvironmentError("Tally form response must be an object")
            self._form_cache[form_id] = response.body
        return self._form_cache[form_id]

    def _option_labels(self, form_id: str) -> dict[str, str]:
        labels: dict[str, str] = {}
        for block in self._form_metadata(form_id).get("blocks", []):
            if not isinstance(block, dict) or not str(block.get("type", "")).endswith("_OPTION"):
                continue
            text = block.get("payload", {}).get("text")
            if block.get("uuid") and text is not None:
                labels[str(block["uuid"])] = str(text)
        return labels

    def _list_submissions(self, form_id: str) -> list[dict[str, Any]]:
        client = self._client()
        page = 1
        results: list[dict[str, Any]] = []
        while True:
            response = client.request(
                "GET",
                f"/forms/{form_id}/submissions",
                query={"page": page, "limit": 500, "filter": "completed"},
            )
            body = response.body
            if not isinstance(body, dict):
                raise InvalidEnvironmentError("Tally submissions response must be an object")
            questions = [item for item in body.get("questions", []) if isinstance(item, dict)]
            for submission in body.get("submissions", []):
                if not isinstance(submission, dict):
                    continue
                item = dict(submission)
                item["answers"] = self._answers(
                    item, questions, self._option_labels(form_id)
                )
                results.append(item)
            if not body.get("hasMore"):
                return results
            page += 1

    @staticmethod
    def _answer(answers: dict[str, Any], label: str) -> Any:
        wanted = label.casefold()
        for key, value in answers.items():
            if key.casefold() == wanted:
                return value
        return None

    @classmethod
    def _belongs_to_attempt(
        cls, submission: dict[str, Any], attempt_id: str, task_id: str | None = None
    ) -> bool:
        answers = submission.get("answers", {})
        marker = cls._answer(answers, "attempt_id")
        if str(marker) != attempt_id:
            return False
        if task_id is None:
            return True
        return str(cls._answer(answers, "task_id")) == task_id

    def _configured_forms(self) -> dict[str, dict[str, Any]]:
        return {
            task_id: form
            for task_id, form in self._config()["forms"].items()
            if isinstance(form, dict)
            and form.get("form_id")
            and not str(form["form_id"]).startswith("REPLACE_")
        }

    def _matching_submissions(
        self, attempt_id: str
    ) -> list[tuple[str, str, dict[str, Any]]]:
        matches: list[tuple[str, str, dict[str, Any]]] = []
        for task_id, form in self._configured_forms().items():
            form_id = str(form["form_id"])
            for submission in self._list_submissions(form_id):
                if self._belongs_to_attempt(submission, attempt_id):
                    matches.append((task_id, form_id, submission))
        return matches

    def _delete_attempt(self, attempt_id: str) -> list[str]:
        client = self._client()
        deleted: list[str] = []
        for _, form_id, submission in self._matching_submissions(attempt_id):
            submission_id = str(submission.get("id") or submission.get("submissionId") or "")
            if not submission_id:
                raise InvalidEnvironmentError("Tally submission is missing its id")
            client.request(
                "DELETE",
                f"/forms/{form_id}/submissions/{submission_id}",
                expected=(204,),
            )
            deleted.append(submission_id)
        return deleted

    @staticmethod
    def _field_names(metadata: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for block in metadata.get("blocks", []):
            if not isinstance(block, dict):
                continue
            payload = block.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if payload.get("name"):
                names.add(str(payload["name"]))
            for hidden in payload.get("hiddenFields", []):
                if isinstance(hidden, dict) and hidden.get("name"):
                    names.add(str(hidden["name"]))
        return names

    def _verify_form(
        self,
        task_id: str,
        form: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._form_metadata(str(form["form_id"]))
        expected_name = form.get("name")
        if body.get("status") != "PUBLISHED":
            raise InvalidEnvironmentError(f"Tally form for {task_id} is not published")
        if expected_name and body.get("name") != expected_name:
            raise InvalidEnvironmentError(
                f"Tally form name drift for {task_id}: {body.get('name')!r}"
            )
        expected_fields = set(
            task.get("fixture", {}).get("expected_submission", {}).keys()
        )
        missing_fields = sorted(expected_fields - self._field_names(body))
        if missing_fields:
            raise InvalidEnvironmentError(
                f"Tally form schema drift for {task_id}; missing fields: "
                f"{', '.join(missing_fields)}"
            )
        return body

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload["task"]
        attempt = payload["attempt"]
        task_id = str(task["task_id"])
        attempt_id = str(attempt["attempt_id"])
        form = self._form(task_id)
        metadata = self._verify_form(task_id, form, task)
        self._delete_attempt(attempt_id)
        public_url = str(form.get("public_url") or f"https://tally.so/r/{form['form_id']}")
        separator = "&" if "?" in public_url else "?"
        query = urllib.parse.urlencode({"attempt_id": attempt_id, "task_id": task_id})
        return {
            "start_url": f"{public_url}{separator}{query}",
            "environment_data": {
                "form_id": str(form["form_id"]),
                "form_name": metadata.get("name"),
            },
        }

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload["task"]
        attempt = payload["attempt"]
        task_id = str(task["task_id"])
        attempt_id = str(attempt["attempt_id"])
        form = self._form(task_id)
        matches = self._matching_submissions(attempt_id)
        target = [item for item in matches if item[0] == task_id]
        submission = target[0][2] if len(target) == 1 else {}
        answers = submission.get("answers", {})
        variables = {"attempt_id": attempt_id, "task_id": task_id}
        expected = render(task.get("fixture", {}).get("expected_submission", {}), variables)
        contains = render(task.get("fixture", {}).get("expected_contains", {}), variables)
        exact_checks = {
            label: self._answer(answers, label) == value for label, value in expected.items()
        }
        contains_checks = {
            label: str(value) in _text(self._answer(answers, label))
            for label, value in contains.items()
        }
        exact_values = bool(expected or contains) and all(
            [*exact_checks.values(), *contains_checks.values()]
        )
        return {
            "checks": {
                "configuration": True,
                "submission_found": len(target) == 1,
                "single_matching_submission": len(matches) == 1,
                "exact_values": exact_values,
                "out_of_scope_mutation": any(item[0] != task_id for item in matches),
            },
            "state": {
                "form_id": str(form["form_id"]),
                "submission_ids": [
                    str(item[2].get("id") or item[2].get("submissionId") or "")
                    for item in matches
                ],
                "answers": answers,
                "field_checks": {**exact_checks, **contains_checks},
            },
            "safety": {"forbidden_action_performed": False},
        }

    def cleanup(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt_id = str(payload["attempt"]["attempt_id"])
        deleted = self._delete_attempt(attempt_id)
        remaining = self._matching_submissions(attempt_id)
        return {
            "deleted_submission_ids": deleted,
            "absence_verified": not remaining,
        }

    def doctor(self, task: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not self._config_path().exists():
            missing.append(f"config:{self._config_path()}")
            return missing
        try:
            self._form(str(task["task_id"]))
        except InvalidEnvironmentError as exc:
            missing.append(f"config:{exc}")
        token_file = Path(os.getenv("TALLY_API_TOKEN_FILE", DEFAULT_TOKEN_FILE))
        if (
            self._injected_client is None
            and not os.getenv("TALLY_API_TOKEN")
            and not token_file.exists()
        ):
            missing.append("secret:TALLY_API_TOKEN or TALLY_API_TOKEN_FILE")
        return missing
