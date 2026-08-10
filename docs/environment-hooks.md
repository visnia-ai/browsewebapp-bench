# Environment lifecycle hooks

Every mutable target uses `prepare`, `observe`, and `cleanup` outside the measured
browser agent. Hooks may use private fixture APIs without exposing those APIs or
credentials to the agent.

The built-in hook command is:

```text
python -m rbbench.integrations.hook ADAPTER PHASE
```

An operator may replace a phase with
`RBBENCH_<ADAPTER>_<PHASE>_CMD`. The command reads `RBBENCH_CONTEXT_FILE` and writes
JSON to `RBBENCH_OUTPUT_FILE`.

## Implemented adapters

| Adapter | Tasks | Lifecycle |
|---|---:|---|
| `public_web` | 20 | Fresh browser context; retained attempt artifacts |
| `ato_simulator` | 8 | Fresh official mock scenario; retained attempt artifacts |
| `tally_public_form` | 6 | Validate form, mark attempt, observe submission, delete, prove absence |
| `controlled_portal` | 5 | Start isolated loopback server, observe state, reset, prove absence, stop |

Tally's trusted hook uses an operator-owned API token. The browser starts on the
published form and never receives that token or the private form-owner session.

The controlled adapter creates one process on `127.0.0.1` and one fictional state
namespace per attempt. Its private observer capability remains in lifecycle
metadata rather than page content.

## Output contract

Prepare may return a new `start_url`, optional session data, and non-secret
`environment_data`. Observe returns the structured evidence required by the task's
oracle. Cleanup for a mutable task must return:

```json
{"absence_verified": true}
```

Missing cleanup proof changes the result to `invalid_environment`; it is not scored
as agent failure.
