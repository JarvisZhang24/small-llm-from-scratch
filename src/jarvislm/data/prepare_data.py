"""Stream text datasets into GPT-2 ``uint16`` training shards."""

import argparse
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from jarvislm.tokenizer import GPT2Tokenizer


class _Tokenizer(Protocol):
    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...


def _close_iterator(iterator: object) -> None:
    """Close a streaming generator before Python begins interpreter shutdown."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class PreparationStats:
    """Summary returned after converting a dataset stream into shards."""

    documents_seen: int
    documents_used: int
    documents_skipped: int
    tokens_written: int
    shards_written: int


class TokenShardWriter:
    """Append GPT-2 token IDs to fixed-size ``uint16`` binary shards."""

    def __init__(self, output_dir: str | Path, shard_size: int) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._buffer = np.empty(shard_size, dtype=np.uint16)
        self._position = 0
        self.shards_written = 0
        self.total_tokens = 0

    def add_tokens(
        self, token_ids: list[int], max_total_tokens: int | None = None
    ) -> None:
        """Append tokens, optionally stopping exactly at a global token budget."""
        if max_total_tokens is not None:
            if max_total_tokens < 0:
                raise ValueError("max_total_tokens must be non-negative")
            token_ids = token_ids[: max(0, max_total_tokens - self.total_tokens)]

        offset = 0
        while offset < len(token_ids):
            take = min(self.shard_size - self._position, len(token_ids) - offset)
            chunk = np.asarray(token_ids[offset : offset + take], dtype=np.int64)
            if chunk.ndim != 1 or (chunk < 0).any() or (chunk >= 2**16).any():
                raise ValueError("token IDs must be one-dimensional uint16 values")

            self._buffer[self._position : self._position + take] = chunk
            self._position += take
            self.total_tokens += take
            offset += take
            if self._position == self.shard_size:
                self.flush()

    def flush(self) -> Path | None:
        """Persist a pending partial shard and return its path, if any."""
        if not self._position:
            return None
        path = self.output_dir / f"shard_{self.shards_written:05d}.bin"
        self._buffer[: self._position].tofile(path)
        self.shards_written += 1
        self._position = 0
        return path


def load_streaming_hf_dataset(
    dataset_name: str,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    hf_token: bool = False,
) -> Iterable[Mapping[str, Any]]:
    """Open a Hugging Face dataset in streaming mode."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Preparing Hugging Face data requires the optional 'datasets' package. "
            "Install the project dependencies before running this command."
        ) from error
    kwargs: dict[str, Any] = {"path": dataset_name, "split": split, "streaming": True}
    if name is not None:
        kwargs["name"] = name
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if hf_token:
        kwargs["token"] = True
    else:
        # ``datasets`` does not load a repository .env itself.  Loading it here
        # lets public FineWeb downloads use HF_TOKEN for higher Hub rate limits
        # without logging or otherwise exposing the secret.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if token := os.environ.get("HF_TOKEN"):
            kwargs["token"] = token
    return load_dataset(**kwargs)


def inspect_dataset(
    dataset_name: str,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    hf_token: bool = False,
    n: int = 3,
) -> list[Mapping[str, Any]]:
    """Return up to ``n`` streaming examples so callers can inspect schemas."""
    if n < 0:
        raise ValueError("n must be non-negative")
    stream = load_streaming_hf_dataset(dataset_name, split, name, data_dir, hf_token)
    examples: list[Mapping[str, Any]] = []
    for example in stream:
        examples.append(example)
        if len(examples) == n:
            break
    return examples


def prepare_streaming_dataset(
    output_dir: str | Path,
    dataset_name: str,
    text_field: str,
    num_tokens: int,
    shard_size: int = 100_000_000,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    hf_token: bool = False,
    min_chars: int = 0,
    *,
    tokenizer: _Tokenizer | None = None,
    stream: Iterable[Mapping[str, Any]] | None = None,
) -> PreparationStats:
    """Tokenize a stream, append GPT-2 EOT per document, and write shards."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")

    tokenizer = tokenizer or GPT2Tokenizer()
    if stream is None:
        stream = load_streaming_hf_dataset(dataset_name, split, name, data_dir, hf_token)
    writer = TokenShardWriter(output_dir, shard_size)
    seen = used = skipped = 0

    iterator = iter(stream)
    try:
        for document in iterator:
            seen += 1
            if text_field not in document:
                raise KeyError(
                    f"text field {text_field!r} is absent; available keys: {list(document)}"
                )
            text = document[text_field]
            if text is None:
                skipped += 1
                continue
            text = str(text)
            if len(text) < min_chars:
                skipped += 1
                continue

            writer.add_tokens(tokenizer.encode(text, add_eos=True), num_tokens)
            used += 1
            if writer.total_tokens == num_tokens:
                break
    finally:
        _close_iterator(iterator)

    writer.flush()
    return PreparationStats(seen, used, skipped, writer.total_tokens, writer.shards_written)


def prepare_fineweb_edu(
    output_dir: str | Path,
    num_tokens: int = 10_000_000_000,
    shard_size: int = 100_000_000,
    sample: str = "sample-10BT",
) -> PreparationStats:
    """Prepare the FineWeb-Edu sample used by the historical V1 project."""
    return prepare_streaming_dataset(
        output_dir, "HuggingFaceFW/fineweb-edu", "text", num_tokens, shard_size, name=sample
    )


def prepare_fineweb_edu_splits(
    train_dir: str | Path,
    val_dir: str | Path,
    *,
    train_tokens: int = 10_000_000_000,
    val_tokens: int = 20_000_000,
    shard_size: int = 100_000_000,
    sample: str = "sample-10BT",
    tokenizer: _Tokenizer | None = None,
    stream: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[PreparationStats, PreparationStats]:
    """Create disjoint FineWeb-Edu validation and training shards in one pass.

    FineWeb-Edu exposes a single ``train`` split.  Reserving complete documents
    for validation before writing training shards prevents data leakage without
    downloading the corpus twice.  The production defaults mirror the 350M
    V1 recipe: ``sample-10BT`` with 10B train tokens in 100M-token
    ``uint16`` shards.  The original repository did not publish its validation
    preparation; JarvisLM reserves a deterministic, disjoint 20M-token split.
    """
    if train_tokens <= 0 or val_tokens <= 0:
        raise ValueError("train_tokens and val_tokens must be positive")

    tokenizer = tokenizer or GPT2Tokenizer()
    stream = stream or load_streaming_hf_dataset(
        "HuggingFaceFW/fineweb-edu", split="train", name=sample
    )
    train_writer = TokenShardWriter(train_dir, shard_size)
    val_writer = TokenShardWriter(val_dir, shard_size)
    seen = used = skipped = 0

    iterator = iter(stream)
    try:
        for document in iterator:
            seen += 1
            text = document.get("text")
            if text is None:
                skipped += 1
                continue
            token_ids = tokenizer.encode(str(text), add_eos=True)
            if not token_ids:
                skipped += 1
                continue
            writer, budget = (
                (val_writer, val_tokens)
                if val_writer.total_tokens < val_tokens
                else (train_writer, train_tokens)
            )
            writer.add_tokens(token_ids, budget)
            used += 1
            if train_writer.total_tokens == train_tokens:
                break
    finally:
        _close_iterator(iterator)

    val_writer.flush()
    train_writer.flush()
    if val_writer.total_tokens != val_tokens or train_writer.total_tokens != train_tokens:
        raise RuntimeError(
            "FineWeb-Edu stream ended before the requested train/validation token budgets"
        )

    val_stats = PreparationStats(seen, used, skipped, val_writer.total_tokens, val_writer.shards_written)
    train_stats = PreparationStats(seen, used, skipped, train_writer.total_tokens, train_writer.shards_written)
    return train_stats, val_stats


def prepare_dummy(
    output_dir: str | Path, num_shards: int = 3, tokens_per_shard: int = 50_000
) -> list[Path]:
    """Create valid random GPT-2 shards for smoke tests, not model training."""
    if num_shards <= 0 or tokens_per_shard <= 0:
        raise ValueError("num_shards and tokens_per_shard must be positive")
    writer = TokenShardWriter(output_dir, tokens_per_shard)
    rng = np.random.default_rng(0)
    for _ in range(num_shards):
        writer.add_tokens(
            rng.integers(0, GPT2Tokenizer.vocab_size, tokens_per_shard).tolist()
        )
    return sorted(Path(output_dir).glob("*.bin"))


def main() -> None:
    """Minimal command-line entry point for preparing pretraining data."""
    parser = argparse.ArgumentParser(description="Write GPT-2 token shards")
    parser.add_argument("mode", choices=("fineweb", "generic", "dummy"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--hf-token", action="store_true")
    parser.add_argument("--val-output-dir", type=Path)
    parser.add_argument("--val-tokens", type=int, default=20_000_000)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--min-chars", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "fineweb":
        if args.val_output_dir is None:
            stats = prepare_fineweb_edu(args.output_dir, args.num_tokens, args.shard_size)
            print(stats)
        else:
            print(
                prepare_fineweb_edu_splits(
                    args.output_dir,
                    args.val_output_dir,
                    train_tokens=args.num_tokens,
                    val_tokens=args.val_tokens,
                    shard_size=args.shard_size,
                )
            )
    elif args.mode == "generic":
        if args.dataset is None:
            parser.error("--dataset is required for generic mode")
        stats = prepare_streaming_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            text_field=args.text_field,
            num_tokens=args.num_tokens,
            shard_size=args.shard_size,
            split=args.split,
            name=args.name,
            data_dir=args.data_dir,
            hf_token=args.hf_token,
            min_chars=args.min_chars,
        )
        print(stats)
    else:
        print(prepare_dummy(args.output_dir))


if __name__ == "__main__":
    main()
