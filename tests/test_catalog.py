from __future__ import annotations

import unittest
from collections import Counter

from rbbench.catalog import BENCHMARK_TASK_IDS, REPO_ROOT, load_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog()

    def test_catalog_has_expected_real_environment_mix(self) -> None:
        self.assertEqual(len(self.catalog.tasks), 100)
        self.assertEqual(
            [task.task_id for task in self.catalog.tasks],
            list(BENCHMARK_TASK_IDS),
        )
        self.assertEqual(
            Counter(task.environment.adapter for task in self.catalog.tasks),
            {
                "ato_simulator": 9,
                "tally_public_form": 6,
                "public_web": 80,
                "controlled_portal": 5,
            },
        )
        synthetic = [
            task
            for task in self.catalog.tasks
            if "synthetic" in task.environment.kind
        ]
        self.assertEqual(len(synthetic), 5)

    def test_every_mutable_task_requires_verified_cleanup(self) -> None:
        mutable = [task for task in self.catalog.tasks if task.environment.mutable]
        self.assertEqual(len(mutable), 11)
        self.assertTrue(all(task.cleanup.verify_absence for task in mutable))
        self.assertTrue(
            all("cleanup" in task.environment.required_hooks for task in mutable)
        )

    def test_tally_tasks_include_attempt_isolation_and_login_variants(self) -> None:
        tally = [
            task
            for task in self.catalog.tasks
            if task.environment.adapter == "tally_public_form"
        ]
        self.assertEqual(len(tally), 6)
        self.assertTrue(
            all(
                task.fixture["expected_submission"]["attempt_id"] == "{{attempt_id}}"
                and task.fixture["expected_submission"]["task_id"] == task.task_id
                for task in tally
            )
        )
        self.assertEqual(
            [task.task_id for task in tally if task.environment.auth == "form_password"],
            ["RBA-010", "RBA-012"],
        )
        self.assertTrue(
            all(
                task.environment.concurrency_key == "tally.so"
                and task.environment.concurrency_limit == 2
                for task in tally
            )
        )

    def test_local_input_artifacts_exist(self) -> None:
        for task in self.catalog.tasks:
            for key in ("input_artifact", "body_artifact", "content_artifact"):
                if key in task.fixture:
                    path = REPO_ROOT / str(task.fixture[key])
                    self.assertTrue(path.exists(), f"{task.task_id}: missing {path}")

    def test_browser_task_text_contains_attempt_scope_when_mutable(self) -> None:
        for task in self.catalog.tasks:
            if task.environment.mutable:
                self.assertIn("{{attempt_id}}", task.confirmed_task, task.task_id)

    def test_download_tasks_do_not_require_agent_side_renaming(self) -> None:
        for task in self.catalog.tasks:
            if "download" not in task.category and "document_download" not in task.category:
                continue
            self.assertNotIn(" as RBA-", task.confirmed_task, task.task_id)
            for assertion in task.oracle.assertions:
                if assertion.kind == "artifact_matches":
                    expected = assertion.expected or {}
                    self.assertNotIn("name", expected, task.task_id)


if __name__ == "__main__":
    unittest.main()
