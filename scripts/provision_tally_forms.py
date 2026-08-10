from __future__ import annotations

import argparse
import json
import sys

from rbbench.integrations.tally_provision import ensure_tally_forms, form_specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision permanent Tally benchmark forms")
    parser.add_argument("--task", action="append", choices=sorted(form_specs()))
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="replace the blocks/settings of already provisioned forms",
    )
    args = parser.parse_args()
    results = ensure_tally_forms(args.task, update_existing=args.update_existing)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
