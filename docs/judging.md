# LLM judgement

The benchmark uses an independent native LLM judge. The browser agent
and judge may use different providers and models. Every executor is judged through
the same evidence contract.

## Evidence supplied

The judge receives, in priority order:

1. Trusted observer state and the trusted artifact inventory.
2. Reviewed reference ground truth, when the task has a `reference_key` and the
   matching reference file is installed.
3. Task text, fixture data, forbidden actions, and the catalog evaluation contract.
4. The agent trajectory and, when `--judge-with-images` is set, up to ten recent
   unique screenshots.
5. The agent's final response.

When a task declares a `reference_key` but the reference file is missing, the
judge still scores the attempt with `reference_ground_truth` unset. That is a
weaker correctness guarantee than a reviewed pack: the judge relies on trusted
observation, the task text, fixture, and trajectory. A missing reference pack
does not make the attempt `invalid_environment`.

The prompt covers task satisfaction, output quality, browser execution,
impossible-task classification, and CAPTCHA handling. It adds explicit evidence
precedence for stateful workflows: a trusted observer failure cannot be overridden
by a persuasive final answer. Exact JSON equality is not required when the result
is semantically complete and correct.

The structured output is:

```json
{
  "reasoning": "evidence-based explanation",
  "verdict": true,
  "failure_reason": "",
  "impossible_task": false,
  "reached_captcha": false,
  "model": "z-ai/glm-5.2",
  "provider": "openai"
}
```

A true verdict scores 1 and a false verdict scores 0. A judge API, dependency, or
structured-output failure makes the attempt `invalid_environment` so evaluator
outages do not count against the browser agent.

## Native built-in judge

The implementation is local to `src/rbbench/judges.py` and uses only Python's
standard library. It imports no external benchmark implementation. Configure the
provider independently of the browser executor:

```bash
pip install -e .
export OPENROUTER_API_KEY=...

rbbench run --name judging-example --task RBA-015 \
  --agent-base-url http://MODEL_HOST:8001/v1
```

With no `--judge-*` overrides, BrowseWebApp bench immediately judges each attempt with
OpenRouter `z-ai/glm-5.2`, pinned to `decart/fp4` with provider fallback disabled,
high reasoning, 39,500 evidence characters, and text-only evidence. This is the
same judge configuration used for the latest catalog semantic-projection Arm C
result. It is a normal inline native judge, not a placeholder followed by rejudge.

Supported built-in provider adapters are Google, OpenAI, and Anthropic. They call
the providers' HTTPS APIs directly. `--judge-base-url` selects a compatible proxy,
and `--judge-api-key-env` selects its API-key environment variable.
`--judge-with-images` attaches screenshots; `--judge-max-images` changes the limit
of ten.

## External judge contract

Use `--judge command --judge-command './my-judge'` to supply another evaluator. The
benchmark writes `judge-input.json` and sets:

| Variable | Meaning |
|---|---|
| `RBBENCH_JUDGE_INPUT_FILE` | Task, ground truth, trusted evidence, trace, and screenshot paths |
| `RBBENCH_JUDGE_OUTPUT_FILE` | Required structured judgement output path |

The command may instead emit the judgement JSON on stdout. This keeps the benchmark
harness-neutral and permits a privately hosted judge model.

## Reproducibility

Each attempt retains `judge-input.json`, `judgement.json`, trusted observation, and
screenshots. Published comparisons must pin judge provider, model, prompt revision,
and catalog version. LLM judgement is intentionally more tolerant than exact code,
but it is not deterministic; model changes and repeated-judge variance should be
reported.
