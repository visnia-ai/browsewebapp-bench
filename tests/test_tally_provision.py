from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rbbench.catalog import Catalog
from rbbench.errors import InvalidEnvironmentError
from rbbench.integrations.tally_provision import (
    TallyApiError,
    ensure_tally_forms,
    form_specs,
)
from rbbench.runner import BenchmarkRunner
from rbbench.schema import RunStatus, TaskRunResult, TaskSpec


class EnsureTallyFormsTests(unittest.TestCase):
    def test_verify_existing_does_not_rewrite_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "forms.json"
            original = {
                "schema_version": 1,
                "provisioned_at": "2026-01-01",
                "forms": {
                    "RBA-009": {
                        "form_id": "form-009",
                        "name": "Operations service intake",
                        "public_url": "https://tally.so/r/form-009",
                    }
                },
            }
            config.write_text(json.dumps(original), encoding="utf-8")
            calls: list[tuple[str, str]] = []

            def request_fn(token, method, path, body=None):
                calls.append((method, path))
                self.assertEqual(token, "test-token")
                self.assertIsNone(body)
                return {"id": "form-009", "name": "Operations service intake"}

            results = ensure_tally_forms(
                ["RBA-009"],
                config_path=config,
                token="test-token",
                request_fn=request_fn,
            )
            self.assertEqual(results["RBA-009"]["action"], "verified_existing")
            self.assertEqual(calls, [("GET", "/forms/form-009")])
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), original)

    def test_create_on_replace_placeholder_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "forms.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provisioned_at": "2026-01-01",
                        "forms": {
                            "RBA-009": {
                                "form_id": "REPLACE_RBA-009",
                                "name": "Operations service intake",
                                "public_url": "",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            def request_fn(token, method, path, body=None):
                self.assertEqual(method, "POST")
                self.assertEqual(path, "/forms")
                self.assertEqual(body, form_specs()["RBA-009"])
                return {"id": "created-009", "name": "Operations service intake"}

            results = ensure_tally_forms(
                ["RBA-009"],
                config_path=config,
                token="test-token",
                request_fn=request_fn,
            )
            self.assertEqual(results["RBA-009"]["action"], "created")
            self.assertEqual(results["RBA-009"]["form_id"], "created-009")
            persisted = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(persisted["forms"]["RBA-009"]["form_id"], "created-009")
            self.assertEqual(
                persisted["forms"]["RBA-009"]["public_url"],
                "https://tally.so/r/created-009",
            )

    def test_recreate_on_failed_get(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "forms.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provisioned_at": "2026-01-01",
                        "forms": {
                            "RBA-009": {
                                "form_id": "missing-009",
                                "name": "Operations service intake",
                                "public_url": "https://tally.so/r/missing-009",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calls: list[str] = []

            def request_fn(token, method, path, body=None):
                calls.append(method)
                if method == "GET":
                    raise TallyApiError("GET", path, 404, "not found")
                self.assertEqual(method, "POST")
                return {"id": "recreated-009", "name": "Operations service intake"}

            results = ensure_tally_forms(
                ["RBA-009"],
                config_path=config,
                token="test-token",
                request_fn=request_fn,
            )
            self.assertEqual(calls, ["GET", "POST"])
            self.assertEqual(results["RBA-009"]["action"], "created")
            persisted = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(persisted["forms"]["RBA-009"]["form_id"], "recreated-009")

    def test_recreate_when_pinned_form_is_not_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "forms.json"
            config.write_text(
                json.dumps(
                    {
                        "forms": {
                            "RBA-009": {
                                "form_id": "inaccessible-009",
                                "name": "Operations service intake",
                                "public_url": "https://tally.so/r/inaccessible-009",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            calls: list[str] = []

            def request_fn(token, method, path, body=None):
                calls.append(method)
                if method == "GET":
                    raise TallyApiError("GET", path, 401, "not authorized")
                self.assertEqual(method, "POST")
                return {"id": "replacement-009", "name": "Operations service intake"}

            results = ensure_tally_forms(
                ["RBA-009"],
                config_path=config,
                token="test-token",
                request_fn=request_fn,
            )

            self.assertEqual(calls, ["GET", "POST"])
            self.assertEqual(results["RBA-009"]["action"], "created")
            persisted = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["forms"]["RBA-009"]["form_id"], "replacement-009"
            )

    def test_non_404_get_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "forms.json"
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
                ),
                encoding="utf-8",
            )

            def request_fn(token, method, path, body=None):
                raise TallyApiError("GET", path, 500, "boom")

            with self.assertRaises(InvalidEnvironmentError):
                ensure_tally_forms(
                    ["RBA-009"],
                    config_path=config,
                    token="test-token",
                    request_fn=request_fn,
                )


def _public_task() -> TaskSpec:
    return TaskSpec.from_dict(
        {
            "task_id": "RBA-999",
            "title": "Runner fixture",
            "confirmed_task": "Return ok.",
            "category": "test",
            "environment": {
                "adapter": "public_web",
                "kind": "production_public_web",
                "start_url": "https://example.com/",
            },
            "fixture": {},
            "oracle": {
                "type": "reference_result",
                "reference_key": "RBA-999",
                "assertions": [
                    {
                        "kind": "equals",
                        "path": "result.primary",
                        "expected_path": "result.primary",
                    }
                ],
            },
            "cleanup": {"strategy": "close_fresh_context"},
            "safety": {},
            "sources": ["https://example.com/"],
        }
    )


def _tally_task() -> TaskSpec:
    return TaskSpec.from_dict(
        {
            "task_id": "RBA-009",
            "title": "Tally fixture",
            "confirmed_task": "Submit the form.",
            "category": "test",
            "environment": {
                "adapter": "tally_public_form",
                "kind": "managed_public_form",
                "start_url": "https://tally.so/r/form-009",
                "concurrency_key": "tally.so",
                "concurrency_limit": 2,
            },
            "fixture": {},
            "oracle": {
                "type": "environment_observation",
                "assertions": [
                    {
                        "kind": "truthy",
                        "path": "checks.submission_found",
                    }
                ],
            },
            "cleanup": {"strategy": "api_delete_attempt"},
            "safety": {},
            "sources": ["https://tally.so/r/form-009"],
        }
    )


class RunnerEnsureHookTests(unittest.TestCase):
    def test_runner_skips_ensure_without_tally_tasks(self) -> None:
        task = _public_task()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog("test", "1", (task,), root / "catalog.json")
            runner = BenchmarkRunner(
                catalog=catalog,
                executor=object(),  # type: ignore[arg-type]
                judge=object(),  # type: ignore[arg-type]
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            stub_result = TaskRunResult(
                task_id=task.task_id,
                attempt_id="attempt",
                status=RunStatus.SUCCESS,
                score=1.0,
            )
            with patch(
                "rbbench.runner.ensure_tally_forms",
                side_effect=AssertionError("should not provision"),
            ):
                with patch.object(
                    runner, "run_one", new=AsyncMock(return_value=stub_result)
                ):
                    summary = asyncio.run(
                        runner.run([task], parallel=1, run_id="no-tally")
                    )
            self.assertEqual(summary["task_count"], 1)

    def test_runner_ensures_forms_when_tally_tasks_selected(self) -> None:
        task = _tally_task()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog("test", "1", (task,), root / "catalog.json")
            runner = BenchmarkRunner(
                catalog=catalog,
                executor=object(),  # type: ignore[arg-type]
                judge=object(),  # type: ignore[arg-type]
                runtime_dir=root / "attempts",
                results_dir=root / "results",
            )
            stub_result = TaskRunResult(
                task_id=task.task_id,
                attempt_id="attempt",
                status=RunStatus.SUCCESS,
                score=1.0,
            )
            with patch("rbbench.runner.ensure_tally_forms") as ensure:
                with patch.object(
                    runner, "run_one", new=AsyncMock(return_value=stub_result)
                ):
                    summary = asyncio.run(
                        runner.run([task], parallel=1, run_id="with-tally")
                    )
            ensure.assert_called_once_with()
            self.assertEqual(summary["task_count"], 1)


if __name__ == "__main__":
    unittest.main()
