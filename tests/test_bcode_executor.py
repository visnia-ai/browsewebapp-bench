from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "bcode_executor.py"
SPEC = importlib.util.spec_from_file_location("rbbench_bcode_executor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BcodeUsageTests(unittest.TestCase):
    def test_openrouter_provider_pin_is_absent_without_explicit_provider(self) -> None:
        with patch.object(MODULE, "MODEL", "openrouter/example/model"), patch.object(
            MODULE, "OPENROUTER_PROVIDER", None
        ):
            self.assertEqual(MODULE.build_openrouter_provider_config(), {})

    def test_openrouter_provider_pin_is_optional_configuration(self) -> None:
        with patch.object(MODULE, "MODEL", "openrouter/example/model"), patch.object(
            MODULE, "OPENROUTER_PROVIDER", "example-provider"
        ):
            config = MODULE.build_openrouter_provider_config()

        model = config["provider"]["openrouter"]["models"]["example/model"]
        self.assertEqual(
            model["options"]["provider"],
            {"order": ["example-provider"], "allow_fallbacks": False},
        )
        self.assertNotIn("limit", model)
        self.assertNotIn("options", config["provider"]["openrouter"])

    def test_step_usage_keeps_cache_and_reasoning_as_subsets(self) -> None:
        usage = MODULE.usage_from_event(
            {
                "type": "step_finish",
                "part": {
                    "tokens": {
                        "input": 100,
                        "output": 30,
                        "reasoning": 70,
                        "cache": {"read": 900, "write": 0},
                        "total": 1100,
                    },
                    "cost": 0.25,
                },
            }
        )

        self.assertEqual(usage["input_tokens"], 1000)
        self.assertEqual(usage["cached_input_tokens"], 900)
        self.assertEqual(usage["output_tokens"], 100)
        self.assertEqual(usage["reasoning_tokens"], 70)
        self.assertEqual(usage["non_reasoning_output_tokens"], 30)
        self.assertEqual(usage["total_tokens"], 1100)
        self.assertEqual(usage["cost"], 0.25)


class BcodeBrowserConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_cdp_url_skips_local_chrome(self) -> None:
        environment = {"BU_CDP_URL": "http://127.0.0.1:9222"}
        launch = AsyncMock()

        with patch.object(MODULE, "launch_chrome", new=launch):
            chrome, profile_dir = await MODULE.connect_browser(environment)

        launch.assert_not_awaited()
        self.assertIsNone(chrome)
        self.assertIsNone(profile_dir)
        self.assertNotIn("BU_CDP_WS", environment)

    async def test_without_external_cdp_launches_isolated_chrome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir = Path(temporary) / "profile"
            chrome = object()
            launch = AsyncMock(
                return_value=(chrome, "ws://127.0.0.1:9222/devtools/browser/test")
            )
            environment: dict[str, str] = {}

            with patch.object(
                MODULE.tempfile, "mkdtemp", return_value=str(profile_dir)
            ), patch.object(MODULE, "launch_chrome", new=launch):
                selected_chrome, selected_profile = await MODULE.connect_browser(
                    environment
                )

        launch.assert_awaited_once_with(profile_dir, "about:blank")
        self.assertIs(selected_chrome, chrome)
        self.assertEqual(selected_profile, profile_dir)
        self.assertEqual(
            environment["BU_CDP_WS"],
            "ws://127.0.0.1:9222/devtools/browser/test",
        )

    async def test_seeded_profile_is_copied_without_live_chrome_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed"
            seed.mkdir()
            (seed / "Default").mkdir()
            (seed / "Default" / "Cookies").write_text("session", encoding="utf-8")
            (seed / "SingletonLock").write_text("lock", encoding="utf-8")
            profile_dir = root / "attempt-profile"
            chrome = object()
            launch = AsyncMock(
                return_value=(chrome, "ws://127.0.0.1:9222/devtools/browser/test")
            )
            environment: dict[str, str] = {}

            with patch.object(
                MODULE.tempfile, "mkdtemp", return_value=str(profile_dir)
            ), patch.object(MODULE, "launch_chrome", new=launch):
                selected_chrome, selected_profile = await MODULE.connect_browser(
                    environment,
                    seed,
                    "https://tally.so/r/test?attempt_id=a&task_id=RBA-009",
                )

            self.assertIs(selected_chrome, chrome)
            self.assertEqual(selected_profile, profile_dir)
            self.assertEqual(
                (profile_dir / "Default" / "Cookies").read_text(encoding="utf-8"),
                "session",
            )
            self.assertFalse((profile_dir / "SingletonLock").exists())
            launch.assert_awaited_once_with(
                profile_dir,
                "https://tally.so/r/test?attempt_id=a&task_id=RBA-009",
            )

    async def test_rejects_ambiguous_external_cdp_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "Set only one"):
            await MODULE.connect_browser(
                {
                    "BU_CDP_URL": "http://127.0.0.1:9222",
                    "BU_CDP_WS": "ws://127.0.0.1:9222/devtools/browser/test",
                }
            )


if __name__ == "__main__":
    unittest.main()
