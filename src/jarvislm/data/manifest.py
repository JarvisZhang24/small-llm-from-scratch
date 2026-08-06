"""Integrity manifests for persistent pretraining shard directories."""

import json
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_FORMAT_VERSION = 1
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SAMPLE = "sample-10BT"
# ``sample-10BT`` is an approximate name.  Streaming the complete current
# sample through the reference GPT-2 tokenizer (including one EOT per document)
# yields this many tokens.  JarvisLM reserves the first 20M for validation and
# uses the remainder for V1 pretraining.
FINEWEB_EDU_V1_TOTAL_TOKENS = 9_953_989_297
FINEWEB_EDU_V1_VAL_TOKENS = 20_000_000
FINEWEB_EDU_V1_TRAIN_TOKENS = FINEWEB_EDU_V1_TOTAL_TOKENS - FINEWEB_EDU_V1_VAL_TOKENS


def count_uint16_tokens(directory: str | Path) -> tuple[int, tuple[Path, ...]]:
    """Return the token count and sorted ``.bin`` shards in ``directory``."""
    directory = Path(directory)
    paths = tuple(sorted(directory.glob("*.bin")))
    if not paths:
        raise FileNotFoundError(f"no .bin shards found in {directory}")

    itemsize = np.dtype(np.uint16).itemsize
    for path in paths:
        if path.stat().st_size % itemsize:
            raise ValueError(f"shard is not a valid uint16 file: {path}")
    return sum(path.stat().st_size // itemsize for path in paths), paths


def build_fineweb_edu_manifest(
    train_dir: str | Path,
    val_dir: str | Path,
    *,
    expected_train_tokens: int,
    expected_val_tokens: int,
) -> dict[str, Any]:
    """Build a manifest only when both persistent split directories are complete."""
    if expected_train_tokens <= 0 or expected_val_tokens <= 0:
        raise ValueError("expected token counts must be positive")

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    train_tokens, train_shards = count_uint16_tokens(train_dir)
    val_tokens, val_shards = count_uint16_tokens(val_dir)
    if train_tokens != expected_train_tokens:
        raise ValueError(
            f"training shards contain {train_tokens:,} tokens; expected "
            f"{expected_train_tokens:,}"
        )
    if val_tokens != expected_val_tokens:
        raise ValueError(
            f"validation shards contain {val_tokens:,} tokens; expected "
            f"{expected_val_tokens:,}"
        )

    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "dataset": {"name": FINEWEB_EDU_DATASET, "config": FINEWEB_EDU_SAMPLE},
        "tokenizer": {"name": "tiktoken:gpt2", "vocab_size": 50_257},
        "dtype": "uint16",
        "splits": {
            "train": {
                "directory": str(train_dir),
                "tokens": train_tokens,
                "shards": [path.name for path in train_shards],
            },
            "val": {
                "directory": str(val_dir),
                "tokens": val_tokens,
                "shards": [path.name for path in val_shards],
            },
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write a preparation manifest without overwriting an existing one."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def verify_fineweb_edu_manifest(
    path: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    *,
    expected_train_tokens: int,
    expected_val_tokens: int,
) -> dict[str, Any]:
    """Validate persisted shards against their manifest and expected token budgets."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing completed-data manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest must contain a JSON object: {path}")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(f"unsupported manifest format: {path}")

    rebuilt = build_fineweb_edu_manifest(
        train_dir,
        val_dir,
        expected_train_tokens=expected_train_tokens,
        expected_val_tokens=expected_val_tokens,
    )
    if manifest != rebuilt:
        raise ValueError(
            "prepared shard contents do not match the manifest; do not train on "
            "this volume until the data preparation is repaired"
        )
    return manifest


__all__ = [
    "FINEWEB_EDU_DATASET",
    "FINEWEB_EDU_SAMPLE",
    "FINEWEB_EDU_V1_TOTAL_TOKENS",
    "FINEWEB_EDU_V1_TRAIN_TOKENS",
    "FINEWEB_EDU_V1_VAL_TOKENS",
    "MANIFEST_FORMAT_VERSION",
    "build_fineweb_edu_manifest",
    "count_uint16_tokens",
    "verify_fineweb_edu_manifest",
    "write_manifest",
]
