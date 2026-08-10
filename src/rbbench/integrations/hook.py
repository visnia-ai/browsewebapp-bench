from __future__ import annotations

import argparse
import sys

from rbbench.errors import BenchmarkError

from .common import context, emit
from .registry import adapter_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a first-party environment phase")
    parser.add_argument("adapter")
    parser.add_argument("phase", choices=("prepare", "observe", "cleanup"))
    args = parser.parse_args(argv)
    adapter = adapter_for(args.adapter)
    try:
        result = getattr(adapter, args.phase)(context())
        emit(result)
        return 0
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
