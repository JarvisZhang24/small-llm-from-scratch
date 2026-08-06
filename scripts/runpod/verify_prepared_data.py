"""Create or verify the FineWeb-Edu manifest stored on a RunPod network volume."""

import argparse
from pathlib import Path

from jarvislm.data import (
    FINEWEB_EDU_V1_TRAIN_TOKENS,
    FINEWEB_EDU_V1_VAL_TOKENS,
    build_fineweb_edu_manifest,
    verify_fineweb_edu_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=FINEWEB_EDU_V1_TRAIN_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=FINEWEB_EDU_V1_VAL_TOKENS)
    parser.add_argument(
        "--write", action="store_true", help="create the manifest after validating shards"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        manifest = build_fineweb_edu_manifest(
            args.data_dir,
            args.val_dir,
            expected_train_tokens=args.train_tokens,
            expected_val_tokens=args.val_tokens,
        )
        path = write_manifest(args.manifest, manifest)
        print(f"wrote completed-data manifest: {path}")
    else:
        verify_fineweb_edu_manifest(
            args.manifest,
            args.data_dir,
            args.val_dir,
            expected_train_tokens=args.train_tokens,
            expected_val_tokens=args.val_tokens,
        )
        print(f"verified completed-data manifest: {args.manifest}")


if __name__ == "__main__":
    main()
