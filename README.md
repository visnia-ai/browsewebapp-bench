# BrowseWebApp bench

BrowseWebApp bench evaluates browser agents on 100 workflows drawn from common
automation work: extracting structured records, downloading or uploading documents,
submitting or creating forms and using complex web apps. By default, the benchmark
uses [Browser Agent](https://github.com/visnia-ai/browser-agent).

## Contents

- [Benchmark results](#benchmark-results)
- [Install](#install)
- [Run](#run)
  - [Browser Agent](#browser-agent)
  - [Bcode](#bcode)
- [Misc](#misc)
- [Secrets](#secrets)
- [License](#license)

## Benchmark results


| Metric               | [Browser Agent](https://github.com/visnia-ai/browser-agent) | [Browsercode](https://github.com/browser-use/browsercode) |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| Model                | gpt-5.6-luna                                                                                              | gpt-5.6-luna |
| Success              | **76%**                                                                                                   | 64%          |
| Duration (seconds)   | **15,653**                                                                                                | 32,036       |
| Cost                 | **$3.73**                                                                                                 | $7.45        |
| Successful tasks / $ | **18.77**                                                                                                 | 8.86         |




## Install

Python 3.11 or newer and Chrome or Chromium are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```



## Run

You can run the benchmark with
[Browser Agent](https://github.com/visnia-ai/browser-agent) or
[browser-code](https://github.com/browser-use/browsercode).

Default judge configuration:

- endpoint: OpenAI API
- model: `gpt-5.6-luna`
- API key: `OPENAI_API_KEY`
- reasoning effort: `high`
- evidence: text-only



### Browser Agent

The default executor uses the pre-installed SDK package for
[Browser Agent](https://github.com/visnia-ai/browser-agent).
Set `<PROVIDER>_API_KEY` for the selected agent provider, such as
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `TOGETHER_API_KEY`.

```bash
export <PROVIDER>_API_KEY=...

rbbench run \
  --name <run-id> \
  --agent-base-url <openai-compatible-base-url> \
  --parallel <N>
```

To benchmark a local [Browser Agent](https://github.com/visnia-ai/browser-agent):

```bash
export <PROVIDER>_API_KEY=...
export BROWSER_AGENT_AUTH_ENCRYPTION_KEY=...

rbbench run \
  --name <run-id> \
  --agent-cli <path-to-browser-agent-cli> \
  --agent-config <path-to.yaml> \
  --parallel <N>
```

Encrypted site logins belong in the YAML as `auth_credentials` (`browser-agent generate-key` / `encrypt`). Plaintext YAML credentials are rejected by Browser
Agent. Prepare-hook session credentials still override YAML when present.

For Codex-backed benchmarks, sign in once before starting the run:

```sh
rbbench codex-login
rbbench run --name <run-id> --provider codex --model <codex-model> --parallel <N>
```



### Bcode

Install with the official script:

```bash
curl -fsSL https://bcode.sh/install | bash
```

Then run the benchmark:

```bash
export OPENROUTER_API_KEY=...
export BCODE_CHROME_BIN=<path-to-chrome-or-chromium>
# optional: BCODE_BIN, BCODE_MODEL, BCODE_MODEL_VARIANT,
#           BCODE_OPENROUTER_PROVIDER

rbbench run \
  --name <run-id> \
  --executor command \
  --executor-command scripts/bcode_executor.py \
  --parallel <N>
```



## Misc

ATO simulator tasks stay sequential while Tally tasks are limited to concurrency 2
by design; other tasks defer to `--parallel`.

Use `--browser-profile /path/to/chrome-user-data` to seed a logged-in Chrome
profile into each attempt. Chrome must not be using that directory at the same
time; the benchmark clones it into an isolated profile for each attempt.

## License

MIT
