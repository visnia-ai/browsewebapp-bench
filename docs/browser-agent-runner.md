# Browser Agent runner

BrowseWebApp bench uses the Browser Agent Python SDK as its default executor. The
published PyPI package `browser-agent-python-sdk` is an unpinned runtime
dependency, so installs resolve to the latest release. Upgrade with
`pip install -U browser-agent-python-sdk`.

```bash
pip install -e .
export OPENROUTER_API_KEY=...
rbbench run \
  --name browser-agent-example \
  --task RBA-015 \
  --agent-base-url http://MODEL_HOST:8001/v1
```

The executor creates an SDK task with the catalog start URL and instruction,
stages upload fixtures under the attempt workspace, assigns downloads to the
attempt artifact directory, and converts the SDK result into the benchmark's
executor result schema. The SDK owns runtime acquisition and browser process
lifecycle.

## Baseline defaults

| Setting | Value |
|---|---|
| provider | `vllm` |
| model | `nvidia/GLM-5.2-NVFP4` |
| reasoning | `high` |
| max model length | `48000` |
| reserved output tokens | `4000` |
| max steps | `50` |
| SDK retries | `0` |
| browser visibility | visible |

The endpoint is intentionally not hard-coded. Pass it with `--agent-base-url`.
Use `--headless` for an invisible browser. `rbbench doctor` verifies that the
SDK is installed and, when PyPI is reachable, that it matches the latest
published version.

The independent judge defaults to OpenAI API `gpt-5.6-luna` with high reasoning
and text-only evidence.
Judge settings remain independent from the Browser Agent executor configuration.

The SDK result does not expose private chain-of-thought. Its terminal task data,
downloaded artifacts, and any workspace screenshots are passed to the independent
judge together with trusted target state and reference ground truth.
