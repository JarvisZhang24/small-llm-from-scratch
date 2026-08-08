"""Score several JarvisLM checkpoints under one identical validation protocol.

The trainer's online validation uses ``eval_batches`` batches of
``micro_batch_size`` sequences, so two runs with different micro-batch sizes
measure different amounts of held-out data.  This script fixes the batch size
and the batch count for every checkpoint it loads, which makes the resulting
losses directly comparable.  By default it consumes the complete validation
split rather than a 20-batch sample.
"""

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from jarvislm.data import PretrainDataset
from jarvislm.model import GPT, ModelConfig

_VARIANT_KEYS = (("model", "raw"), ("ema", "EMA"))


def _strip_compile_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name.removeprefix("_orig_mod."): tensor for name, tensor in state_dict.items()}


def _load_variants(
    path: Path, device: torch.device
) -> tuple[ModelConfig, int | None, dict[str, GPT]]:
    """Return every scoreable weight set stored in one checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping payload: {path}")
    config = ModelConfig(**dict(checkpoint["model_config"]))

    variants: dict[str, GPT] = {}
    for key, label in _VARIANT_KEYS:
        if key not in checkpoint:
            continue
        model = GPT(config)
        model.load_state_dict(_strip_compile_prefix(checkpoint[key]))
        variants[label] = model.to(device).eval()
    if not variants:
        raise ValueError(f"checkpoint contains no model weights: {path}")
    return config, checkpoint.get("step"), variants


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int]:
    """Return mean next-token loss and the number of batches consumed."""
    total_loss = torch.zeros((), dtype=torch.float64)
    batches = 0
    for index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and index == max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss: {loss}")
        total_loss += loss.double().cpu()
        batches += 1
        if batches % 100 == 0:
            print(f"  ... {batches} batches", flush=True)
    if not batches:
        raise ValueError("validation loader produced no complete batches")
    return (total_loss / batches).item(), batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare checkpoints on one identical validation protocol"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        metavar="LABEL=PATH",
        help="e.g. V1=runs/v1/checkpoints/last.pt V2=runs/v2/checkpoints/last.pt",
    )
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="fixed for every checkpoint so the measured data is identical",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="default: the complete validation split",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write results and protocol settings to this JSON file",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive when provided")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    entries: list[tuple[str, Path]] = []
    for item in args.checkpoints:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        entries.append((label, Path(path)))

    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    seq_len: int | None = None
    results: list[tuple[str, int | None, float, int]] = []

    for label, path in entries:
        config, step, variants = _load_variants(path, device)
        if loader is None:
            seq_len = config.max_seq_len
            dataset = PretrainDataset(args.val_dir, seq_len)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            print(
                f"validation split: {len(dataset):,} sequences of {seq_len} tokens; "
                f"batch size {args.batch_size}"
            )
        elif config.max_seq_len != seq_len:
            raise ValueError(
                f"{label} uses max_seq_len={config.max_seq_len}, expected {seq_len}"
            )

        for variant, model in variants.items():
            name = f"{label} {variant}"
            print(f"evaluating {name} (step {step})", flush=True)
            loss, batches = evaluate(model, loader, device, args.max_batches)
            results.append((name, step, loss, batches * args.batch_size))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    measured = {sequences for *_, sequences in results}
    if len(measured) != 1:
        raise RuntimeError(f"checkpoints saw different sequence counts: {measured}")
    sequences = measured.pop()
    print(
        f"\nEvaluated every checkpoint on the same {sequences:,} sequences "
        f"({args.batch_size} sequences/batch).\n"
    )
    print("| Checkpoint | Step | Validation loss | Perplexity |")
    print("| --- | ---: | ---: | ---: |")
    for name, step, loss, _ in results:
        step_text = "n/a" if step is None else f"{step:,}"
        print(f"| {name} | {step_text} | {loss:.4f} | {math.exp(loss):.2f} |")

    if args.output is not None:
        report = {
            "protocol": {
                "validation_dir": str(args.val_dir),
                "sequence_length": seq_len,
                "batch_size": args.batch_size,
                "sequences_evaluated": sequences,
                "tokens_evaluated": sequences * (seq_len or 0),
                "max_batches": args.max_batches,
                "device": str(device),
            },
            "checkpoints": {label: str(path) for label, path in entries},
            "results": [
                {
                    "name": name,
                    "step": step,
                    "validation_loss": loss,
                    "perplexity": math.exp(loss),
                }
                for name, step, loss, _ in results
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
