"""Emit a machine-readable verdict for an A100 350M preflight run."""

import argparse
import json
from pathlib import Path

from jarvislm.training.preflight import (
    assess_preflight,
    checkpoint_step,
    parse_training_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=500)
    parser.add_argument("--min-metric-records", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1_000.0)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed_step = checkpoint_step(args.checkpoint) if args.checkpoint.is_file() else None
    report = assess_preflight(
        parse_training_metrics(args.log),
        completed_step=completed_step,
        expected_step=args.expected_step,
        min_metric_records=args.min_metric_records,
        max_grad_norm=args.max_grad_norm,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise SystemExit("A100 preflight failed; do not start the H200 full run")


if __name__ == "__main__":
    main()
