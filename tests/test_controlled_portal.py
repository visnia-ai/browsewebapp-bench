from __future__ import annotations

import base64
import json
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

from rbbench.catalog import load_catalog
from rbbench.integrations.controlled import ControlledPortalIntegration


class ControlledPortalE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self, task_id: str) -> tuple[ControlledPortalIntegration, dict, str]:
        attempt_dir = self.root / task_id
        (attempt_dir / "artifacts").mkdir(parents=True)
        task = load_catalog().by_id(task_id).to_dict()
        attempt = {
            "attempt_id": f"e2e-{task_id.lower()}",
            "task_id": task_id,
            "attempt_dir": str(attempt_dir),
            "artifact_dir": str(attempt_dir / "artifacts"),
            "environment_data": {},
        }
        integration = ControlledPortalIntegration()
        prepared = integration.prepare({"task": task, "attempt": attempt})
        attempt["environment_data"].update(prepared["environment_data"])
        return integration, {"task": task, "attempt": attempt}, str(prepared["start_url"])

    @staticmethod
    def request(url: str, *, form: dict | None = None, json_body: dict | None = None):
        data = None
        headers = {}
        if form is not None:
            data = urllib.parse.urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            finally:
                exc.close()

    def assert_observed_and_clean(self, integration, context):
        observation = integration.observe(context)
        self.assertTrue(observation["checks"]["configuration"])
        self.assertTrue(observation["checks"]["behavior"])
        self.assertEqual(integration.cleanup(context), {"absence_verified": True})

    def test_all_controlled_workflows_have_trusted_state_and_cleanup(self) -> None:
        integration, context, start = self.context("RBA-046")
        credentials = context["task"]["fixture"]["credentials"]
        credentials["email"] = credentials["email"].replace("{{attempt_id}}", "e2e-rba-046")
        self.assertEqual(self.request(start, form=credentials)[0], 200)
        self.assert_observed_and_clean(integration, context)

        integration, context, start = self.context("RBA-047")
        self.assertEqual(self.request(start)[0], 200)
        base = context["attempt"]["environment_data"]["controlled_base_url"]
        self.assertEqual(self.request(base + "/records/attachment")[0], 200)
        self.assertEqual(self.request(base + "/records/edit", form={})[0], 403)
        self.assert_observed_and_clean(integration, context)

        integration, context, start = self.context("RBA-048")
        base = context["attempt"]["environment_data"]["controlled_base_url"]
        query = "status=Exception&from=2026-06-01&to=2026-06-30"
        for page in (1, 2, 3):
            self.assertEqual(self.request(f"{base}/vendor/invoices?{query}&page={page}")[0], 200)
        self.assertEqual(self.request(base + "/vendor/export.csv")[0], 200)
        self.assertEqual(self.request(base + "/vendor/export.pdf")[0], 200)
        self.assert_observed_and_clean(integration, context)

        integration, context, start = self.context("RBA-049")
        base = context["attempt"]["environment_data"]["controlled_base_url"]
        document = context["task"]["fixture"]["document"]
        artifact = (
            Path(__file__).resolve().parents[1]
            / context["task"]["fixture"]["input_artifact"]
        ).read_bytes()
        status, body = self.request(
            base + "/cases/CASE-1049/upload",
            json_body={
                "name": document["filename"],
                "size": len(artifact),
                "data_base64": base64.b64encode(artifact).decode("ascii"),
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "Accepted")
        page_status, page_body = self.request(start)
        self.assertEqual(page_status, 200)
        self.assertIn(b"<strong id=workflow-status>Accepted</strong>", page_body)
        self.assert_observed_and_clean(integration, context)

        integration, context, start = self.context("RBA-050")
        base = context["attempt"]["environment_data"]["controlled_base_url"]
        self.assertEqual(self.request(start)[0], 200)
        stale = {"version": "1", "status": "Resolved", "note": "Verified after refreshing the conflicting update."}
        self.assertEqual(self.request(base + "/records/conflict", form=stale)[0], 200)
        fresh = {**stale, "version": "2"}
        self.assertEqual(self.request(base + "/records/conflict", form=fresh)[0], 200)
        self.assert_observed_and_clean(integration, context)


if __name__ == "__main__":
    unittest.main()
