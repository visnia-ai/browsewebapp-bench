from __future__ import annotations

import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from rbbench.executors import (
    BrowserUseExecutor,
    _OpenAIClientProxy,
    _clone_chrome_profile,
    _install_openai_usage_tracker,
    _start_browser_at,
)


class BrowserUseUsageTests(unittest.TestCase):
    def test_tracks_reasoning_subset_without_changing_browser_use_usage(self) -> None:
        calls = []

        class LLM:
            def _get_usage(self, response):
                calls.append(response)
                return "browser-use-usage"

        llm = LLM()
        totals = _install_openai_usage_tracker(llm)
        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(
                completion_tokens_details=types.SimpleNamespace(
                    reasoning_tokens=73
                )
            )
        )

        self.assertEqual(llm._get_usage(response), "browser-use-usage")
        self.assertEqual(calls, [response])
        self.assertEqual(totals, {"reasoning_tokens": 73, "responses": 1})

    def test_openrouter_provider_is_pinned_without_fallback(self) -> None:
        executor = BrowserUseExecutor(
            model="z-ai/glm-5.2",
            provider="openai",
            openrouter_provider="baseten",
        )
        self.assertEqual(executor.openrouter_provider, "baseten")

    def test_profile_clone_omits_live_chrome_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "seed"
            source.mkdir()
            (source / "Default").mkdir()
            (source / "Default" / "Cookies").write_text("session", encoding="utf-8")
            (source / "SingletonLock").write_text("lock", encoding="utf-8")
            clone = _clone_chrome_profile(source, prefix="rbbench-test-bu-")
            try:
                self.assertEqual(
                    (clone / "Default" / "Cookies").read_text(encoding="utf-8"),
                    "session",
                )
                self.assertFalse((clone / "SingletonLock").exists())
            finally:
                import shutil

                shutil.rmtree(clone, ignore_errors=True)


class OpenAIRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_starts_at_exact_prepared_url(self) -> None:
        browser = AsyncMock()
        url = "https://tally.so/r/test?attempt_id=a&task_id=RBA-009"

        await _start_browser_at(browser, url)

        browser.start.assert_awaited_once_with()
        browser.navigate_to.assert_awaited_once_with(url)

    async def test_proxy_merges_baseten_pin_into_request_body(self) -> None:
        calls = []

        async def create(*args, **kwargs):
            calls.append((args, kwargs))
            return "response"

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )
        )
        proxy = _OpenAIClientProxy(
            client,
            {
                "provider": {
                    "only": ["baseten"],
                    "allow_fallbacks": False,
                }
            },
        )

        result = await proxy.chat.completions.create(
            model="z-ai/glm-5.2", extra_body={"trace": "kept"}
        )

        self.assertEqual(result, "response")
        self.assertEqual(
            calls,
            [
                (
                    (),
                    {
                        "model": "z-ai/glm-5.2",
                        "extra_body": {
                            "trace": "kept",
                            "provider": {
                                "only": ["baseten"],
                                "allow_fallbacks": False,
                            },
                        },
                    },
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
