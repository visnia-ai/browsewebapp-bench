from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rbbench.judges import (
    NativeLLMAdapter,
    _JUDGE_SCHEMA,
    _json_object,
    _image_parts,
    construct_judge_prompt,
    judge_evidence,
    load_reference,
)
from rbbench.io import write_json
from rbbench.schema import ExecutionResult

from test_runner import public_task


class JudgeTests(unittest.TestCase):
    def test_reference_is_loaded_as_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"result": {"primary": "42", "details": {"unit": "widgets"}}}
            write_json(root / "RBA-999.json", expected)
            self.assertEqual(load_reference(public_task(), root), expected)

    def test_missing_reference_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(load_reference(public_task(), Path(temporary)))

    def test_prompt_contains_trusted_and_reference_evidence(self) -> None:
        task = public_task()
        execution = ExecutionResult(final_result="forty two", steps=["opened page"])
        evidence = judge_evidence(
            task,
            execution,
            {"checks": {"behavior": True}, "artifacts": []},
            {"result": {"primary": "42"}},
        )
        system, user = construct_judge_prompt(evidence, 0)
        self.assertIn("Trusted observer state", system)
        self.assertIn('"primary": "42"', user)
        self.assertIn("forty two", user)

    def test_recent_screenshots_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one.png"
            duplicate = root / "two.png"
            latest = root / "three.jpg"
            first.write_bytes(b"same")
            duplicate.write_bytes(b"same")
            latest.write_bytes(b"latest")
            images, omitted = _image_parts(
                [str(first), str(duplicate), str(latest)],
                max_images=10,
                text_only=False,
            )
            self.assertEqual(len(images), 2)
            self.assertEqual(images[-1][0], "image/jpeg")
            self.assertEqual(omitted, [])

    def test_screenshot_outside_attempt_roots_is_not_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "attempt"
            allowed.mkdir()
            outside = root / "private.png"
            outside.write_bytes(b"private")
            images, omitted = _image_parts(
                [str(outside)],
                max_images=10,
                text_only=False,
                allowed_roots=(allowed,),
            )
            self.assertEqual(images, [])
            self.assertEqual(omitted, [str(outside)])

    def test_native_output_parser_requires_real_booleans(self) -> None:
        parsed = _json_object(
            '{"reasoning":"ok","verdict":true,"failure_reason":"",'
            '"impossible_task":false,"reached_captcha":false}'
        )
        self.assertTrue(parsed["verdict"])
        with self.assertRaisesRegex(Exception, "verdict"):
            _json_object(
                '{"reasoning":"bad","verdict":"false","failure_reason":"",'
                '"impossible_task":false,"reached_captcha":false}'
            )

    def test_native_output_parser_normalizes_null_success_failure_reason(self) -> None:
        parsed = _json_object(
            '{"reasoning":"complete","verdict":true,"failure_reason":null,'
            '"impossible_task":false,"reached_captcha":false}'
        )
        self.assertEqual(parsed["failure_reason"], "")

    def test_native_output_parser_extracts_json_after_reasoning_text(self) -> None:
        parsed = _json_object(
            '<think>checked the evidence</think>\n'
            '{"reasoning":"complete","verdict":true,"failure_reason":"",'
            '"impossible_task":false,"reached_captcha":false}'
        )
        self.assertTrue(parsed["verdict"])

    def test_native_openai_adapter_builds_multimodal_request(self) -> None:
        adapter = NativeLLMAdapter(
            provider="openai",
            model="judge-model",
            api_key="test-key",
            reasoning_effort="high",
            max_output_tokens=1234,
            request_extra_body={
                "provider": {"only": ["decart/fp4"], "allow_fallbacks": False}
            },
        )
        captured = {}

        def fake_request(payload, headers):
            captured["payload"] = payload
            captured["headers"] = headers
            return {
                "choices": [{"message": {"content": (
                    '{"reasoning":"supported","verdict":true,'
                    '"failure_reason":"","impossible_task":false,'
                    '"reached_captcha":false}'
                )}}]
            }

        adapter._request = fake_request
        output = adapter._openai("system", "user", [("image/png", "aW1hZ2U=")])
        self.assertIn("json_schema", captured["payload"]["response_format"])
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertEqual(captured["payload"]["max_tokens"], 1234)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            captured["payload"]["provider"],
            {"only": ["decart/fp4"], "allow_fallbacks": False},
        )
        self.assertIn("data:image/png;base64", str(captured["payload"]))
        self.assertIn('"verdict":true', output)

    def test_native_google_adapter_uses_json_schema_and_inline_image(self) -> None:
        adapter = NativeLLMAdapter(
            provider="google", model="judge-model", api_key="test-key"
        )
        captured = {}

        def fake_request(payload, headers):
            captured["payload"] = payload
            captured["headers"] = headers
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}

        adapter._request = fake_request
        adapter._google("system", "user", [("image/jpeg", "aW1hZ2U=")])
        config = captured["payload"]["generationConfig"]
        self.assertEqual(config["responseJsonSchema"], _JUDGE_SCHEMA)
        self.assertEqual(captured["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(
            captured["payload"]["contents"][0]["parts"][1]["inlineData"]["mimeType"],
            "image/jpeg",
        )

    def test_native_anthropic_adapter_uses_native_image_block(self) -> None:
        adapter = NativeLLMAdapter(
            provider="anthropic", model="judge-model", api_key="test-key"
        )
        captured = {}

        def fake_request(payload, headers):
            captured["payload"] = payload
            captured["headers"] = headers
            return {"content": [{"type": "text", "text": "{}"}]}

        adapter._request = fake_request
        adapter._anthropic("system", "user", [("image/webp", "aW1hZ2U=")])
        source = captured["payload"]["messages"][0]["content"][1]["source"]
        self.assertEqual(source["media_type"], "image/webp")
        self.assertEqual(captured["headers"]["x-api-key"], "test-key")

    def test_native_adapter_calls_http_endpoint_without_provider_sdk(self) -> None:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                requests.append(
                    {
                        "authorization": self.headers["Authorization"],
                        "body": self.rfile.read(length).decode("utf-8"),
                    }
                )
                response = {
                    "choices": [{"message": {"content": (
                        '{"reasoning":"local endpoint","verdict":true,'
                        '"failure_reason":"","impossible_task":false,'
                        '"reached_captcha":false}'
                    )}}]
                }
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = NativeLLMAdapter(
                provider="openai",
                model="local-model",
                api_key="local-key",
                base_url=f"http://127.0.0.1:{server.server_port}/judge",
            )
            result = asyncio.run(adapter.invoke("system", "user", []))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(result["verdict"])
        self.assertEqual(requests[0]["authorization"], "Bearer local-key")
        self.assertIn('"model": "local-model"', requests[0]["body"])

    def test_native_adapter_retries_transient_rate_limit(self) -> None:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(self.path)
                if len(requests) == 1:
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                response = {
                    "choices": [{"message": {"content": (
                        '{"reasoning":"retried","verdict":true,'
                        '"failure_reason":"","impossible_task":false,'
                        '"reached_captcha":false}'
                    )}}]
                }
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = NativeLLMAdapter(
                provider="openai",
                model="local-model",
                api_key="local-key",
                base_url=f"http://127.0.0.1:{server.server_port}/judge",
            )
            result = asyncio.run(adapter.invoke("system", "user", []))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(result["verdict"])
        self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
