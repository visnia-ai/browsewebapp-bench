#!/usr/bin/env python3
"""Validate the installed SDK against BrowseWebApp bench's default agent options."""

from __future__ import annotations

import argparse
import importlib.metadata
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("browser-agent-python-sdk")

    from browser_agent import BrowserAgent

    with tempfile.TemporaryDirectory(prefix="browser-bench-sdk-smoke-") as root:
        agent = BrowserAgent(
            provider="vllm",
            model="nvidia/GLM-5.2-NVFP4",
            download_directory=str(Path(root) / "downloads"),
            reasoning_effort="high",
            endpoint_url=args.endpoint,
            max_model_len=48_000,
            reserve_output_tokens=4_000,
            headless=False,
            max_steps=50,
            concurrency=1,
            runs_per_task=1,
            retry_count=0,
        )
        options = agent._options
        expected = {
            "provider": "vllm",
            "model": "nvidia/GLM-5.2-NVFP4",
            "reasoning_effort": "high",
            "max_model_len": 48_000,
            "reserve_output_tokens": 4_000,
            "headless": False,
            "max_steps": 50,
        }
        actual = {key: getattr(options, key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"SDK normalized unexpected options: {actual}")
    print(f"browser-agent-python-sdk {version}: baseline options accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
