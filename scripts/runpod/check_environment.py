"""Validate RunPod-injected secrets and hardware without printing secret values."""

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "full"), required=True)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_secret(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"missing required environment variable: {name}; attach the RunPod Secret "
            "to this Pod before starting the workflow"
        )


def main() -> None:
    args = parse_args()
    require_secret("HF_TOKEN")
    has_wandb_token = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN"))
    if args.use_wandb and not has_wandb_token:
        raise SystemExit(
            "missing WANDB_API_KEY or WANDB_TOKEN; attach the RunPod Secret, or "
            "set JARVISLM_USE_WANDB=0"
        )

    gpu_name: str | None = None
    if args.stage != "prepare":
        try:
            import torch
        except ImportError as error:
            raise SystemExit("PyTorch is required to validate the Pod GPU") from error
        if not torch.cuda.is_available():
            raise SystemExit(f"{args.stage} requires a CUDA GPU Pod")
        gpu_name = torch.cuda.get_device_name(0)
        expected = "A100" if args.stage == "preflight" else "H200"
        if expected not in gpu_name.upper():
            raise SystemExit(f"{args.stage} requires an NVIDIA {expected}; found: {gpu_name}")

    print(
        {
            "stage": args.stage,
            "hf_token_loaded": True,
            "wandb_token_loaded": has_wandb_token,
            "gpu": gpu_name,
        }
    )


if __name__ == "__main__":
    main()
