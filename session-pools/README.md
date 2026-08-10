# Private lifecycle state

`session-pools/private/` is gitignored. The managed Tally lifecycle can read its
API token from `private/tally-api-token`; the file should be owner-readable only.
Hooks use the token directly and never copy it into task JSON, executor inputs,
traces, or result files.
