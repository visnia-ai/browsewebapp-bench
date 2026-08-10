# Reference packs

Public databases, calculators, and official simulators change independently of this
repository. Their reviewed expected values therefore live in a versioned reference
pack rather than being guessed in the task catalog. The LLM judge compares results
semantically against this evidence instead of requiring byte-for-byte or object-shape
equality. Tally tasks do not use this pack: their mutable result is established by
trusted provider state.

A missing reference file no longer blocks scoring. The judge proceeds with
`reference_ground_truth` unset and evaluates from trusted observation, task text,
fixture, and trajectory. Install the pack whenever you need reviewed expected
facts; without it, correctness checks are weaker and `rbbench doctor` warns
rather than marking the task unready.

## Capture

1. Run the fixed task manually in a clean context on the pinned locale, viewport, and release date.
2. Produce the same structured observation contract expected from a trusted executor.
3. Review exact values, links, artifact text, and safety state with a second person.
4. Install it:

```bash
rbbench capture-reference RBA-015 /path/to/reviewed-observation.json \
  --reference-dir references/tasks
```

For the common result oracle, a reference resembles:

```json
{
  "_meta": {
    "captured_at": "2026-07-27T12:00:00Z",
    "target_version": "observed build or page hash",
    "locale": "en-GB",
    "viewport": "1440x900",
    "reviewers": ["operator-1", "operator-2"]
  },
  "result": {
    "primary": "canonical terminal classification or value",
    "details": {"field": "canonical structured value"}
  }
}
```

Reviewed reference files are included for the current release. A release pack may be signed or checksummed and distributed separately when legal/policy review requires it.

## Health runs

Before every model batch, a known-good browser implementation reruns at least one canonical fixture per target family. If the reference run fails, tasks from that family are quarantined as `invalid_environment`. Never update expected output solely because a model disagrees.

## Document artifacts

Artifact observers should parse rather than visually guess:

- PDF MIME, page count, normalized text, required/forbidden strings, section order, and coarse layout signature.
- CSV header order, row count, exact normalized cells, and encoding.
- PNG dimensions, MIME, and QR decoded destination rather than a brittle full-file hash.

Hashes are appropriate only for invariant benchmark input assets. Production-generated files often include timestamps or metadata and should be checked structurally.
