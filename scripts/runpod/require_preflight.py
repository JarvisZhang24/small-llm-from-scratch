"""Fail closed unless an A100 preflight report explicitly passed."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text())
    except FileNotFoundError as error:
        raise SystemExit(f"missing A100 preflight report: {args.report}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid A100 preflight report: {args.report}") from error

    if not isinstance(report, dict) or report.get("passed") is not True:
        raise SystemExit("A100 preflight did not pass; refusing to start the full run")
    if report.get("expected_step") != args.expected_step:
        raise SystemExit(
            "A100 preflight used a different required step count; refusing to start "
            "the full run"
        )
    print(f"A100 preflight passed: {args.report}")


if __name__ == "__main__":
    main()
