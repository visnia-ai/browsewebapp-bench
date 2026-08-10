# Tally managed lifecycle

## What is permanent

The benchmark operator owns one private Tally account and six published forms. The
forms are pinned in `configs/tally/forms.json`; PDF inputs are checked into
`fixtures/tally/`. No harness runner needs a Tally account or authenticated browser
session.

| Task | Form | Variation |
|---|---|---|
| RBA-009 | Service intake | Standard typed submission |
| RBA-010 | Incident report | Password gate and conditional-looking incident data |
| RBA-011 | Compliance certificate | PDF upload and metadata |
| RBA-012 | Solicitud de soporte | Spanish UI plus password gate |
| RBA-013 | Purchase request | Line-item quantities and total reconciliation |
| RBA-014 | Invoice intake | Transcription from a supplied PDF |

Form account credentials are one-time administrative inputs. They are deliberately
not read by the runner. The trusted lifecycle requires only an API token capable of
reading and deleting submissions and reading form metadata.

## Attempt protocol

`TallyIntegration.prepare` validates that the selected form is still published with
the expected name. It removes any stale submission carrying the same attempt ID and
returns a URL such as:

```text
https://tally.so/r/FORM_ID?attempt_id=RUN-RBA-009&task_id=RBA-009
```

Both values populate permanent hidden fields. They are server-side isolation keys,
not model-reported claims. The observer scans all six forms, requires exactly one
submission for the attempt, checks that it belongs to the intended task, and compares
every answer with the rendered fixture. This also catches accidental submission of a
different benchmark form.

Cleanup deletes every matching submission across all configured forms and then
queries again. If any match remains, `absence_verified` is false and the runner
changes the outcome to `invalid_environment`.

This permits concurrent attempts on a single form because attempt IDs are globally
unique. Overall concurrency is governed by `--parallel`; ATO simulator tasks remain
sequential (`concurrency_limit` 1) as the catalog exception.

## Secrets

Use one of:

```bash
export TALLY_API_TOKEN='...'
export TALLY_API_TOKEN_FILE='/absolute/private/path/tally-api-token'
```

The managed local default is the ignored file
`session-pools/private/tally-api-token`, mode `0600`. The token is used only in the
prepare, observe, and cleanup subprocesses. It is never added to `task.json`,
`attempt.json`, executor environment variables, or model instructions.

## One-time provisioning or repair

Benchmark runs that include any `tally_public_form` task call
`ensure_tally_forms` once at start. That verifies pinned IDs in
`configs/tally/forms.json`, creates any missing forms (`REPLACE_*` placeholders or
404s), and persists new IDs back to the config. Normal runs never rewrite form
definitions for forms that already exist.

The checked-in provisioner owns the complete form specification and deterministic
block UUIDs. Use it only as an administrative operation after an intentional form
change:

```bash
python scripts/provision_tally_forms.py --update-existing
```

Review the resulting `configs/tally/forms.json`, smoke-test the rendered forms,
and bump the catalog version.

Regenerate the two fictional PDF inputs with:

```bash
python scripts/build_tally_fixtures.py
```

The JSON sidecars record their canonical text fields. Generation is deterministic at
the content level; the benchmark observer grades uploaded filename/content metadata,
not a fragile PDF byte hash.

## Operational checks

```bash
rbbench doctor --adapter tally_public_form
python -m unittest tests.test_integrations -v
```

Before a scored batch, submit one canary using a unique attempt ID, verify that the
observer sees the exact values, run cleanup, and confirm a second observation finds
no submission. Rotate an API key by creating its replacement first, updating the
private secret store, running doctor and the canary, and only then revoking the old
key.
