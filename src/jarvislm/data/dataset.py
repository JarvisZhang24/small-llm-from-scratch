"""Datasets for uint16 GPT-2 pretraining shards."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class _ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Index fixed-length next-token examples across memory-mapped shards."""

    def __init__(self, shard_paths: Iterable[str | Path], seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.seq_len = seq_len
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        if not self.shard_paths:
            raise ValueError("at least one shard is required")

        self.shards: list[np.memmap] = []
        cumulative_chunks: list[int] = []
        total_chunks = 0
        for path in self.shard_paths:
            if not path.is_file():
                raise FileNotFoundError(f"shard does not exist: {path}")
            if path.stat().st_size % np.dtype(np.uint16).itemsize:
                raise ValueError(f"shard is not a valid uint16 file: {path}")

            shard = np.memmap(path, dtype=np.uint16, mode="r")
            shard_chunks = max(0, (len(shard) - 1) // seq_len)
            total_chunks += shard_chunks
            self.shards.append(shard)
            cumulative_chunks.append(total_chunks)

        if total_chunks == 0:
            raise ValueError("shards do not contain one complete training sequence")

        self._cumulative_chunks = np.asarray(cumulative_chunks, dtype=np.int64)
        self._length = total_chunks

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} is out of range for {self._length} chunks")

        shard_index = int(np.searchsorted(self._cumulative_chunks, index, side="right"))
        previous_chunks = (
            int(self._cumulative_chunks[shard_index - 1]) if shard_index else 0
        )
        start = (index - previous_chunks) * self.seq_len
        tokens = self.shards[shard_index][start : start + self.seq_len + 1]
        tokens = torch.from_numpy(tokens.astype(np.int64))
        return tokens[:-1], tokens[1:]


class PretrainDataset(_ShardDataset):
    """Load every ``*.bin`` GPT-2 ``uint16`` shard in one directory.

    A sample is ``(input_ids, targets)``; both tensors have shape ``[seq_len]``
    and targets are shifted forward by one token.
    """

    def __init__(self, data_dir: str | Path, seq_len: int = 1024) -> None:
        directory = Path(data_dir)
        shard_paths = sorted(directory.glob("*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"no .bin shards found in {directory}")
        super().__init__(shard_paths, seq_len)


__all__ = ["PretrainDataset"]
