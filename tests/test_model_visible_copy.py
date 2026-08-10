from __future__ import annotations

import json
import unittest

from rbbench.catalog import load_catalog
from rbbench.integrations.controlled_server import page
from rbbench.integrations.tally_provision import form_specs


FORBIDDEN_DISCLOSURES = (
    "benchmark fixture",
    "benchmark-only",
    "benchmark mailbox",
    "fictional",
    "rbbench",
)


class ModelVisibleCopyTests(unittest.TestCase):
    def assert_no_disclosures(self, value: object) -> None:
        rendered = json.dumps(value, ensure_ascii=False).lower()
        for phrase in FORBIDDEN_DISCLOSURES:
            self.assertNotIn(phrase, rendered)

    def test_tally_form_definitions_use_realistic_copy(self) -> None:
        self.assert_no_disclosures(form_specs())
        for spec in form_specs().values():
            title = spec["blocks"][0]["payload"]["title"]
            self.assertFalse(title.lower().startswith("rbb"))

    def test_controlled_portal_shell_does_not_identify_the_benchmark(self) -> None:
        self.assert_no_disclosures(page("Account", "Signed in").decode())

    def test_tally_and_controlled_prompts_do_not_use_fixture_language(self) -> None:
        tasks = [
            task.to_dict()
            for task in load_catalog().tasks
            if task.environment.adapter
            in {"tally_public_form", "controlled_portal"}
        ]
        self.assertTrue(tasks)
        for task in tasks:
            self.assert_no_disclosures(task["confirmed_task"])


if __name__ == "__main__":
    unittest.main()
