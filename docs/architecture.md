# Architecture

The benchmark owns task definitions, attempt IDs, lifecycle coordination,
artifacts, ground truth, independent judging, and result classification. Provider
secrets and private operator state remain outside the catalog and source tree.

```text
catalog -> prepare -> isolated browser attempt -> executor -> trusted observer
        -> independent LLM judge -> cleanup -> absence proof -> result
```

## Executors

The default `BrowserAgentExecutor` calls the installed Browser Agent Python SDK.
Each attempt receives its own workspace and download directory. Local input files
are copied into that workspace before the model starts. The SDK returns terminal
task data; the executor also inventories generated images and downloaded files.

`CommandExecutor` remains a harness-neutral extension point. It starts one trusted
subprocess per task and passes:

| Variable | Meaning |
|---|---|
| `RBBENCH_TASK_FILE` | Expanded task JSON |
| `RBBENCH_ATTEMPT_FILE` | Attempt ID, start URL, artifacts, and prepared session handoff |
| `RBBENCH_OUTPUT_FILE` | Path where the harness writes its result JSON |
| `RBBENCH_ARTIFACT_DIR` | Attempt-scoped directory for downloads and evidence |

The result schema is:

```json
{
  "final_result": "human-readable terminal response",
  "steps": ["optional trace summaries"],
  "screenshots": [],
  "observation": {
    "result": {"primary": "structured terminal value", "details": {}},
    "page": {"url": "https://..."},
    "safety": {"forbidden_action_performed": false}
  },
  "metrics": {"steps": 12, "duration_seconds": 48.2, "cost": 0.04},
  "error": null
}
```

Screenshot paths must resolve inside the attempt directory. The judge rejects
other paths so an executor cannot cause arbitrary local files to be uploaded.

## Environment boundary

Read-only public targets need no setup hook. Tally and controlled-portal tasks use
the same three phases:

- `prepare`: create or validate attempt state and resolve the start URL.
- `observe`: read trusted state independently of the agent's claims.
- `cleanup`: remove attempt state and prove absence.

Setup and observation may use privileged APIs because they are benchmark fixture
operations, not actions available to the measured agent.

## Judge

The native multimodal judge receives the task, fixture, evaluation contract,
reviewed reference, trusted observer output, artifact inventory, terminal answer,
available trace summaries, and recent unique screenshots. Trusted state and
reviewed references take priority over agent claims. Semantic correctness matters;
incidental formatting does not.

Judge infrastructure failures are `invalid_environment`. An external judge command
can replace the native adapter without changing the browser executor.

## Safety, secrets, and concurrency

- The catalog contains no provider API key, reusable cookie, account password,
  private OTP seed, or reset token.
- Only fictional mutable records and `example.invalid` identities are used.
- Legal submissions, payments, purchases, external messages, and cross-tenant
  mutations are forbidden.
- One global concurrency gate is combined with per-target semaphores.
- Each controlled attempt uses a separate loopback process and state namespace.
