from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rbbench.catalog import load_catalog
from rbbench.environments import HookedEnvironment
from rbbench.integrations.common import HttpResponse
from rbbench.integrations.tally import TallyIntegration


class FakeTallyClient:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    def request(self, method, path, *, query=None, body=None, expected=(200,)):
        if method == "GET" and path == "/forms/form-009":
            return HttpResponse(
                200,
                {
                    "id": "form-009",
                    "name": "Operations service intake",
                    "status": "PUBLISHED",
                    "blocks": [
                        {
                            "type": "HIDDEN_FIELDS",
                            "payload": {
                                "hiddenFields": [
                                    {"name": "attempt_id"},
                                    {"name": "task_id"},
                                ]
                            },
                        },
                        {"type": "INPUT_TEXT", "payload": {"name": "Requester name"}},
                    ],
                },
                {},
            )
        if method == "GET" and path == "/forms/form-009/submissions":
            questions = [
                {"id": "q-attempt", "title": "attempt_id"},
                {"id": "q-task", "title": "task_id"},
                {"id": "q-name", "title": "Requester name"},
            ]
            return HttpResponse(
                200,
                {
                    "page": 1,
                    "limit": 500,
                    "hasMore": False,
                    "questions": questions,
                    "submissions": list(self.submissions),
                },
                {},
            )
        if method == "DELETE" and path.startswith("/forms/form-009/submissions/"):
            submission_id = path.rsplit("/", 1)[-1]
            self.submissions = [item for item in self.submissions if item["id"] != submission_id]
            return HttpResponse(204, None, {})
        raise AssertionError((method, path, query, body, expected))


def context_for(task_id: str, root: Path) -> dict:
    task = load_catalog().by_id(task_id).to_dict()
    attempt = {
        "attempt_id": "integration-attempt",
        "task_id": task_id,
        "start_url": task["environment"]["start_url"],
        "attempt_dir": str(root / "attempt"),
        "artifact_dir": str(root / "attempt" / "artifacts"),
        "session": {},
        "environment_data": {},
    }
    Path(attempt["artifact_dir"]).mkdir(parents=True)
    return {"task": task, "attempt": attempt}


class IntegrationLifecycleTests(unittest.TestCase):
    def test_tally_submission_is_graded_then_deleted_by_attempt_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tally-forms.json"
            config.write_text(
                json.dumps(
                    {
                        "forms": {
                            "RBA-009": {
                                "form_id": "form-009",
                                "name": "Operations service intake",
                                "public_url": "https://tally.so/r/form-009",
                            }
                        }
                    }
                )
            )
            payload = {
                "task": {
                    "task_id": "RBA-009",
                    "fixture": {
                        "expected_submission": {
                            "attempt_id": "{{attempt_id}}",
                            "task_id": "RBA-009",
                            "Requester name": "Casey Test",
                        }
                    },
                },
                "attempt": {"attempt_id": "attempt-009"},
            }
            client = FakeTallyClient()
            integration = TallyIntegration(client)
            with patch.dict(os.environ, {"TALLY_FORMS_CONFIG": str(config)}):
                prepared = integration.prepare(payload)
                self.assertEqual(
                    prepared["start_url"],
                    "https://tally.so/r/form-009?attempt_id=attempt-009&task_id=RBA-009",
                )
                client.submissions.append(
                    {
                        "id": "submission-1",
                        "responses": [
                            {"questionId": "q-attempt", "answer": "attempt-009"},
                            {"questionId": "q-task", "answer": "RBA-009"},
                            {"questionId": "q-name", "answer": "Casey Test"},
                        ],
                    }
                )
                observed = integration.observe(payload)
                self.assertTrue(observed["checks"]["submission_found"])
                self.assertTrue(observed["checks"]["exact_values"])
                cleaned = integration.cleanup(payload)
                self.assertEqual(cleaned["deleted_submission_ids"], ["submission-1"])
                self.assertTrue(cleaned["absence_verified"])

    def test_hooked_environment_uses_builtin_integration_by_default(self) -> None:
        task = load_catalog().by_id("RBA-009")
        with tempfile.TemporaryDirectory() as temporary:
            environment = HookedEnvironment(Path(temporary))
            with patch.dict(os.environ, {}, clear=True):
                command = environment._command(task, "prepare")
            self.assertIn("rbbench.integrations.hook", command or "")
            self.assertIn("tally_public_form", command or "")


if __name__ == "__main__":
    unittest.main()
