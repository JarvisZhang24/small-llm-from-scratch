"""Single-H100 training entry point for the JarvisLM-350M baseline.

This module intentionally contains no DDP, multi-node, SFT, or 1.5B route.
The default configuration reproduces the reference project's 350M architecture
and optimization recipe, while its step count targets approximately 10B tokens.
"""

import argparse
import copy
import math
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jarvislm.data import PretrainDataset, prepare_fineweb_edu_splits
from jarvislm.model import GPT, ModelConfig
from jarvislm.optim.muon import configure_optimizers

TOKENS_PER_REFERENCE_STEP = 16 * 32 * 1_024
TARGET_TRAIN_TOKENS = 10_000_000_000


def _reference_model_config() -> ModelConfig:
    """The original project's 350M, single-stream pretraining architecture."""
    return ModelConfig(
        vocab_size=50_304,
        max_seq_len=1_024,
        d_model=1_024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=False,
        use_diff_attn=False,
        use_mhc=False,
    )


@dataclass
class TrainConfig:
    """Configuration for one 350M run on one NVIDIA H100 GPU."""

    model: ModelConfig = field(default_factory=_reference_model_config)
    run_name: str = "jarvislm-350m"
    data_dir: Path = Path("data/fineweb-edu/train")
    val_dir: Path = Path("data/fineweb-edu/val")

    # 16 * 32 * 1,024 = 524,288 tokens/update, as in the reference recipe.
    micro_batch_size: int = 16
    grad_accumulation_steps: int = 32
    max_steps: int = math.ceil(TARGET_TRAIN_TOKENS / TOKENS_PER_REFERENCE_STEP)

    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    muon_learning_rate: float = 1.5e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    use_muon: bool = True

    use_ema: bool = True
    ema_decay: float = 0.9995
    warmup_steps: int = 1_000
    compile_model: bool = True
    device: str = "cuda"
    require_h100: bool = True
    seed: int = 42
    num_workers: int = 4

    checkpoint_dir: Path | None = Path("checkpoints/jarvislm-350m")
    save_interval: int = 1_000
    keep_last_checkpoints: int = 3
    resume: bool = True

    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 20
    eval_use_ema: bool = False
    use_wandb: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("350M training supports device='cuda'; cpu is only for smoke tests")
        if self.micro_batch_size <= 0 or self.grad_accumulation_steps <= 0:
            raise ValueError("micro_batch_size and grad_accumulation_steps must be positive")
        if self.max_steps <= 0 or self.warmup_steps < 0:
            raise ValueError("max_steps must be positive and warmup_steps non-negative")
        if self.max_learning_rate <= 0 or self.min_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate must not exceed max_learning_rate")
        if self.muon_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("muon_learning_rate must be positive and weight_decay non-negative")
        if self.grad_clip_norm <= 0 or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("grad_clip_norm must be positive and ema_decay must be in [0, 1)")
        if self.num_workers < 0 or self.save_interval < 0 or self.keep_last_checkpoints < 1:
            raise ValueError("worker/checkpoint intervals must be non-negative; keep_last_checkpoints >= 1")
        if self.log_interval <= 0 or self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("logging and evaluation intervals must be positive")

    @property
    def tokens_per_step(self) -> int:
        return self.model.max_seq_len * self.micro_batch_size * self.grad_accumulation_steps

    @classmethod
    def smoke(cls, device: str = "cpu") -> "TrainConfig":
        """Small CPU-safe integration configuration; it is not a 350M run."""
        return cls(
            model=ModelConfig(
                vocab_size=128,
                max_seq_len=16,
                d_model=64,
                n_layers=1,
                n_heads=4,
                n_kv_heads=4,
                ffn_hidden_dim=128,
                use_flash=False,
                use_qk_norm=False,
                use_diff_attn=False,
                use_mhc=False,
            ),
            run_name="smoke",
            micro_batch_size=2,
            grad_accumulation_steps=1,
            max_steps=2,
            max_learning_rate=1e-3,
            min_learning_rate=1e-3,
            muon_learning_rate=5e-4,
            warmup_steps=0,
            compile_model=False,
            device=device,
            require_h100=False,
            num_workers=0,
            checkpoint_dir=None,
            save_interval=0,
            eval_interval=100,
            use_wandb=False,
        )


@dataclass(frozen=True)
class TrainMetrics:
    """One completed optimizer update."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens: int
    tokens_per_second: float


@dataclass(frozen=True)
class TrainResult:
    device: str
    parameter_count: int
    metrics: tuple[TrainMetrics, ...]
    elapsed_seconds: float


class EMA:
    """Weight-only exponential moving average used for optional validation."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(self.model.parameters(), model.parameters()):
            ema_parameter.lerp_(parameter, 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)


class _SyntheticPretrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic next-token data used only by the CPU smoke test."""

    def __init__(self, model: ModelConfig, examples: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(
            model.vocab_size, (examples, model.max_seq_len + 1), generator=generator
        )
        self.input_ids, self.targets = tokens[:, :-1], tokens[:, 1:]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.targets[index]


def cosine_learning_rate(config: TrainConfig, step: int) -> float:
    """Reference schedule: linear warmup followed by cosine decay."""
    if step < config.warmup_steps:
        return config.min_learning_rate + (
            (config.max_learning_rate - config.min_learning_rate)
            * step
            / max(config.warmup_steps, 1)
        )
    if step >= config.max_steps:
        return config.min_learning_rate
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.min_learning_rate + 0.5 * (
        config.max_learning_rate - config.min_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _require_device(config: TrainConfig) -> torch.device:
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the 350M H100 run but is not available")
        device_name = torch.cuda.get_device_name(0)
        if config.require_h100 and "H100" not in device_name.upper():
            raise RuntimeError(f"This run requires an NVIDIA H100, found: {device_name}")
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], config: TrainConfig, *, shuffle: bool
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if len(dataset) < config.micro_batch_size:
        raise ValueError("dataset is smaller than micro_batch_size")
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=config.device == "cuda",
    )


def _batches_forever(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], count: int
) -> Iterable[list[tuple[torch.Tensor, torch.Tensor]]]:
    iterator = iter(loader)
    while True:
        batches = []
        for _ in range(count):
            try:
                batches.append(next(iterator))
            except StopIteration:
                iterator = iter(loader)
                batches.append(next(iterator))
        yield batches


def _set_learning_rates(
    adamw: torch.optim.Optimizer,
    muon: torch.optim.Optimizer | None,
    config: TrainConfig,
    step: int,
) -> float:
    learning_rate = cosine_learning_rate(config, step)
    for group in adamw.param_groups:
        group["lr"] = learning_rate
    if muon is not None:
        scaled_muon_lr = learning_rate * config.muon_learning_rate / config.max_learning_rate
        for group in muon.param_groups:
            group["lr"] = scaled_muon_lr
    return learning_rate


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batches: int,
) -> float:
    """Return mean next-token loss for a fixed number of validation batches."""
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    try:
        for index, (inputs, targets) in enumerate(loader):
            if index == batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return validation loss")
            losses.append(loss.float().cpu())
    finally:
        model.train(was_training)
    if not losses:
        raise ValueError("validation loader has no complete batches")
    return torch.stack(losses).mean().item()


def _checkpoint_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("step_*.pt")) if directory.exists() else []


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    paths = _checkpoint_paths(directory)
    final_checkpoint = directory / "last.pt"
    if final_checkpoint.is_file():
        paths.append(final_checkpoint)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    adamw: torch.optim.Optimizer,
    muon: torch.optim.Optimizer | None,
    ema: EMA | None,
    step: int,
    config: TrainConfig,
) -> Path:
    """Write all state required to continue a single-GPU run."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "step": step,
        "model": model.state_dict(),
        "adamw": adamw.state_dict(),
        "model_config": asdict(config.model),
        "cpu_rng_state": torch.random.get_rng_state(),
    }
    if muon is not None:
        payload["muon"] = muon.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _prune_checkpoints(directory: Path, keep: int) -> None:
    for old_path in _checkpoint_paths(directory)[:-keep]:
        old_path.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    adamw: torch.optim.Optimizer,
    muon: torch.optim.Optimizer | None,
    ema: EMA | None,
    device: torch.device,
) -> int:
    """Restore model, optimizers, EMA, and RNG state; return completed steps."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    adamw.load_state_dict(checkpoint["adamw"])
    if muon is not None and "muon" in checkpoint:
        muon.load_state_dict(checkpoint["muon"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    torch.random.set_rng_state(checkpoint["cpu_rng_state"])
    if device.type == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint["step"])


def _start_wandb(config: TrainConfig):
    if not config.use_wandb:
        return None
    try:
        import wandb
        from dotenv import load_dotenv
    except ImportError as error:
        raise ImportError("W&B logging requires wandb and python-dotenv") from error
    # Loads .env without printing its contents; WANDB_TOKEN/WANDB_API_KEY stays local.
    load_dotenv()
    if "WANDB_API_KEY" not in os.environ and "WANDB_TOKEN" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
    return wandb.init(
        project="jarvislm-350m",
        name=config.run_name,
        config={
            "model": asdict(config.model),
            "tokens_per_step": config.tokens_per_step,
            "max_steps": config.max_steps,
            "use_muon": config.use_muon,
            "use_ema": config.use_ema,
        },
    )


def _prepare_data_if_requested(
    config: TrainConfig,
    enabled: bool,
    train_tokens: int,
    val_tokens: int,
) -> None:
    if not enabled:
        return
    if config.data_dir.exists() and any(config.data_dir.glob("*.bin")):
        raise FileExistsError(f"refusing to overwrite existing training shards: {config.data_dir}")
    if config.val_dir.exists() and any(config.val_dir.glob("*.bin")):
        raise FileExistsError(f"refusing to overwrite existing validation shards: {config.val_dir}")
    prepare_fineweb_edu_splits(
        config.data_dir,
        config.val_dir,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )


def train(
    config: TrainConfig,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None,
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> TrainResult:
    """Run the 350M loop. Caller-supplied datasets are used by unit tests only."""
    config.validate()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    device = _require_device(config)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    raw_model = GPT(config.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    ema = EMA(raw_model, config.ema_decay) if config.use_ema else None
    muon, adamw = (
        configure_optimizers(
            raw_model,
            lr=config.max_learning_rate,
            muon_lr=config.muon_learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.use_muon
        else (
            None,
            torch.optim.AdamW(
                raw_model.parameters(),
                lr=config.max_learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            ),
        )
    )
    model: nn.Module = torch.compile(raw_model) if config.compile_model else raw_model

    if dataset is None:
        dataset = PretrainDataset(config.data_dir, config.model.max_seq_len)
    train_loader = _make_loader(dataset, config, shuffle=True)
    if val_dataset is None and config.val_dir.exists():
        val_dataset = PretrainDataset(config.val_dir, config.model.max_seq_len)
    val_loader = _make_loader(val_dataset, config, shuffle=False) if val_dataset else None

    start_step = 0
    if config.resume and config.checkpoint_dir is not None:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            start_step = load_checkpoint(latest, raw_model, adamw, muon, ema, device)
            print(f"Resumed from {latest} at completed step {start_step}")

    print(
        f"JarvisLM-350M | parameters={parameter_count:,} | device={device} | "
        f"tokens/update={config.tokens_per_step:,} | target steps={config.max_steps:,}"
    )
    run = _start_wandb(config)
    started = time.perf_counter()
    metrics: list[TrainMetrics] = []
    try:
        for step, microbatches in zip(
            range(start_step, config.max_steps),
            _batches_forever(train_loader, config.grad_accumulation_steps),
        ):
            step_started = time.perf_counter()
            learning_rate = _set_learning_rates(adamw, muon, config, step)
            adamw.zero_grad(set_to_none=True)
            if muon is not None:
                muon.zero_grad(set_to_none=True)

            total_loss = 0.0
            for inputs, targets in microbatches:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    _, loss = model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss: {loss}")
                (loss / config.grad_accumulation_steps).backward()
                total_loss += loss.detach().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm: {grad_norm.item()}")
            if muon is not None:
                muon.step()
            adamw.step()
            if ema is not None:
                ema.update(raw_model)

            completed_step = step + 1
            duration = time.perf_counter() - step_started
            metric = TrainMetrics(
                step=completed_step,
                loss=total_loss / config.grad_accumulation_steps,
                learning_rate=learning_rate,
                grad_norm=float(grad_norm.item()),
                tokens=config.tokens_per_step,
                tokens_per_second=config.tokens_per_step / max(duration, 1e-9),
            )
            metrics.append(metric)

            if completed_step % config.log_interval == 0:
                print(
                    f"step {completed_step:>6d} | loss {metric.loss:.4f} | "
                    f"lr {learning_rate:.2e} | grad {metric.grad_norm:.2f} | "
                    f"tok/s {metric.tokens_per_second:,.0f}"
                )
                if run is not None:
                    run.log({"train/loss": metric.loss, "train/lr": learning_rate,
                             "train/grad_norm": metric.grad_norm,
                             "train/tokens_per_second": metric.tokens_per_second}, step=completed_step)

            if val_loader is not None and completed_step % config.eval_interval == 0:
                eval_model = ema.model if config.eval_use_ema and ema is not None else raw_model
                val_loss = evaluate(eval_model, val_loader, device, config.eval_batches)
                print(f"validation step {completed_step:>6d} | loss {val_loss:.4f}")
                if run is not None:
                    run.log({"validation/loss": val_loss}, step=completed_step)

            if config.save_interval and completed_step % config.save_interval == 0:
                path = save_checkpoint(
                    config.checkpoint_dir / f"step_{completed_step:06d}.pt",
                    raw_model,
                    adamw,
                    muon,
                    ema,
                    completed_step,
                    config,
                )
                _prune_checkpoints(config.checkpoint_dir, config.keep_last_checkpoints)
                print(f"saved checkpoint: {path}")
    finally:
        if config.checkpoint_dir is not None and metrics:
            final_path = save_checkpoint(
                config.checkpoint_dir / "last.pt",
                raw_model,
                adamw,
                muon,
                ema,
                start_step + len(metrics),
                config,
            )
            print(f"saved final checkpoint: {final_path}")
        if run is not None:
            run.finish()
    return TrainResult(device.type, parameter_count, tuple(metrics), time.perf_counter() - started)


def smoke_test(device: str = "cpu") -> TrainResult:
    config = TrainConfig.smoke(device)
    examples = config.micro_batch_size * (config.max_steps + 1)
    return train(config, _SyntheticPretrainDataset(config.model, examples, config.seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-H100 JarvisLM-350M recipe")
    parser.add_argument("--smoke", action="store_true", help="two CPU-safe integration updates")
    parser.add_argument("--prepare-data", action="store_true", help="stream FineWeb-Edu shards before training")
    parser.add_argument("--prepare-only", action="store_true", help="prepare shards, then exit without training")
    parser.add_argument("--prepare-train-tokens", type=int, default=TARGET_TRAIN_TOKENS)
    parser.add_argument("--prepare-val-tokens", type=int, default=20_000_000)
    parser.add_argument("--run-name", default="jarvislm-350m")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fineweb-edu/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/fineweb-edu/val"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/jarvislm-350m"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-h100", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.prepare_data and (
        args.prepare_train_tokens <= 0 or args.prepare_val_tokens <= 0
    ):
        parser.error("--prepare-train-tokens and --prepare-val-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.prepare_only and not args.prepare_data:
        raise ValueError("--prepare-only requires --prepare-data")
    if args.smoke:
        result = smoke_test()
    else:
        config = TrainConfig(
            run_name=args.run_name,
            data_dir=args.data_dir,
            val_dir=args.val_dir,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.wandb,
            compile_model=args.compile,
            resume=args.resume,
            require_h100=args.require_h100,
        )
        for key in ("max_steps", "micro_batch_size", "grad_accumulation_steps", "num_workers"):
            value = getattr(args, key)
            if value is not None:
                setattr(config, key, value)
        config.validate()
        _prepare_data_if_requested(
            config,
            args.prepare_data,
            args.prepare_train_tokens,
            args.prepare_val_tokens,
        )
        if args.prepare_only:
            print("FineWeb-Edu shard preparation completed")
            return
        result = train(config)
    print(
        f"completed {len(result.metrics)} updates on {result.device} in "
        f"{result.elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
