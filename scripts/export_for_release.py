"""Export inference-only weights from a JarvisLM training checkpoint.

A training checkpoint carries optimizer state, EMA shadow weights, and RNG
state so a run can resume; that is several times larger than the weights
themselves and is not useful to someone who only wants to run the model. This
script writes a release directory holding one weight set plus the model
configuration needed to rebuild it.

Weight tying: when ``tie_weights`` is set, ``lm_head.weight`` and
``token_embeddings.weight`` are the same tensor. ``safetensors`` refuses to
serialize shared storage, so the duplicate key is dropped on export and
reconstructed at load time by ``GPT.__init__``.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from jarvislm.model import GPT, ModelConfig

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

_LOAD_SNIPPET = '''import json
from pathlib import Path

{loader_import}

from jarvislm.model import GPT, ModelConfig

fields = ModelConfig.__dataclass_fields__
metadata = json.loads(Path("config.json").read_text())
config = ModelConfig(**{{k: v for k, v in metadata.items() if k in fields}})

state = {loader_call}
if config.tie_weights:
    # The LM head shares the embedding table, so it is not stored separately.
    state["lm_head.weight"] = state["token_embeddings.weight"]

model = GPT(config).eval()
model.load_state_dict(state)
'''

_LOADERS = {
    "model.safetensors": (
        "from safetensors.torch import load_file",
        'load_file("model.safetensors")',
    ),
    "pytorch_model.bin": (
        "import torch",
        'torch.load("pytorch_model.bin", map_location="cpu")',
    ),
}


def _strip_compile_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name.removeprefix("_orig_mod."): tensor for name, tensor in state.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export release weights from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("raw", "ema"),
        default="raw",
        help="'ema' requires the checkpoint to carry EMA shadow weights",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument(
        "--label", help="human-readable run name recorded in config.json"
    )
    parser.add_argument(
        "--require-safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail instead of falling back to a pickled pytorch_model.bin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"checkpoint must contain a mapping payload: {args.checkpoint}")

    key = "model" if args.variant == "raw" else "ema"
    if key not in checkpoint:
        raise SystemExit(
            f"checkpoint has no '{key}' weights; available: {sorted(checkpoint)}"
        )
    config = ModelConfig(**dict(checkpoint["model_config"]))
    state = _strip_compile_prefix(checkpoint[key])

    # Rebuilding the model proves the exported tensors actually load before
    # anything is published.
    model = GPT(config)
    model.load_state_dict(state)
    parameters = sum(p.numel() for p in model.parameters())

    dtype = _DTYPES[args.dtype]
    state = {name: tensor.to(dtype) for name, tensor in model.state_dict().items()}
    if config.tie_weights:
        state.pop("lm_head.weight", None)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        weights_path = output_dir / "model.safetensors"
        save_file(state, weights_path)
    except ImportError:
        if args.require_safetensors:
            raise SystemExit(
                "safetensors is not installed; run `pip install safetensors` or "
                "pass --no-require-safetensors to fall back to a pickled .bin"
            ) from None
        weights_path = output_dir / "pytorch_model.bin"
        torch.save(state, weights_path)
        print(
            "warning: safetensors is not installed, so the weights were pickled "
            "to pytorch_model.bin. Hugging Face flags pickled weights in the UI; "
            "`pip install safetensors` and re-export to publish model.safetensors."
        )

    # A stale file from an earlier export in the other format would otherwise be
    # uploaded alongside this one and load differently.
    for name in _LOADERS:
        stale = output_dir / name
        if name != weights_path.name and stale.exists():
            stale.unlink()
            print(f"removed stale {stale}")

    metadata = asdict(config) | {
        "architecture": "JarvisLM-GPT",
        "weights": args.variant,
        "dtype": args.dtype,
        "parameters": parameters,
        "training_step": checkpoint.get("step"),
        "source_checkpoint": args.checkpoint.name,
    }
    if args.label:
        metadata["run"] = args.label
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    size_gb = weights_path.stat().st_size / 1024**3
    print(f"wrote {weights_path} ({size_gb:.2f} GiB, {args.dtype})")
    print(f"wrote {output_dir / 'config.json'}")
    print(f"parameters: {parameters:,} | variant: {args.variant} | step: {checkpoint.get('step')}")
    loader_import, loader_call = _LOADERS[weights_path.name]
    print("\nLoad with:\n")
    print(_LOAD_SNIPPET.format(loader_import=loader_import, loader_call=loader_call))


if __name__ == "__main__":
    main()
