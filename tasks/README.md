# Task catalog

`tasks.json` contains the 100 fixed task definitions plus shared environment,
safety, and oracle profiles. The loader expands profiles before passing a task to
an executor or lifecycle hook, so `rbbench show RBA-009` displays the normalized
task object.

Within a catalog release, fixture values are immutable. Variants should preserve
the browser-specific capability while changing a real branch or structure, such as
status filters, document layouts, date boundaries, validation paths, localization,
pagination, or bounded state transitions.
