# ATO and IRS release verification — 2026-07-29

RBA-001–008 and RBA-027–028 were exercised against the live production-domain
simulator and calculator interfaces. Independent judgement used the native judge
with `nvidia/GLM-5.2-NVFP4`, high reasoning effort, text-only projected evidence,
and a 39,500-character evidence guard.

## Positive runs

| Run ID | Tasks | Result |
|---|---|---|
| `verified-ato-candidates-20260729` | RBA-001–008 | 8 success, 0 failure, mean score 1.0 |
| `verified-irs-candidates-20260729` | RBA-027–028 | 2 success, 0 failure, mean score 1.0 |

The earlier canonical-capture runs used separate disposable browser contexts and
were reviewed to construct the reference files. The final positive runs used new
contexts and the corrected task definitions.

## Negative judge checks

`scripts/check_negative_judgements.py` evaluated an empty/incomplete execution and
a plausible-navigation execution with materially wrong values for every task. The
report at `.runs/negative-judge-checks-ato-irs-20260729.json` records 20 of 20
expected rejections.

## Live-schema corrections

- RBA-001 now asks whether values came from the visible pre-filled item rather than
  requesting a nonexistent per-item status.
- RBA-003 requests GST 1A/1B and PAYG income-tax instalment 5A, matching the
  activity-statement detail pages.
- RBA-004 requests all four displayed account transactions rather than five.
- RBA-005 removes the ambiguous account-level effective date and requests the
  effective date for each transaction.
- RBA-006 requests the fields actually rendered by the current employment record.
- RBA-007 names the rendered ENCC determination and amount fields explicitly.
- RBA-008 requests all five visible Subject/Channel/Issue Date rows and does not
  ask for nonexistent income-year or read-status columns.
- RBA-027 and RBA-028 specify every required personal, employment, deduction, and
  credit branch. RBA-028 requests the exact rendered W-4 entries rather than an
  inferred per-pay-period reduction.

The release references produced by those runs are stored under
`references/tasks/` and remain subject to the reference-health process described
in `docs/reference-packs.md`.
