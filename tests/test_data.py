import numpy as np
import pytest

from jarvislm.data import (
    PretrainDataset,
    TokenShardWriter,
    prepare_fineweb_edu_splits,
    prepare_streaming_dataset,
)


class FakeTokenizer:
    eos_id = 99

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        token_ids = [len(text)]
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids


def test_writer_and_pretrain_dataset_cross_shard_boundary(tmp_path) -> None:
    writer = TokenShardWriter(tmp_path, shard_size=4)
    writer.add_tokens(list(range(8)))

    dataset = PretrainDataset(tmp_path, seq_len=2)
    assert len(dataset) == 2
    input_ids, targets = dataset[1]
    assert input_ids.tolist() == [4, 5]
    assert targets.tolist() == [5, 6]
    with pytest.raises(IndexError):
        dataset[-1]


def test_writer_rejects_token_ids_outside_uint16(tmp_path) -> None:
    writer = TokenShardWriter(tmp_path, shard_size=4)
    with pytest.raises(ValueError, match="uint16"):
        writer.add_tokens([65_536])


def test_prepare_streaming_dataset_appends_eot_and_honors_token_budget(tmp_path) -> None:
    stats = prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="unused-with-injected-stream",
        text_field="text",
        num_tokens=3,
        shard_size=10,
        tokenizer=FakeTokenizer(),
        stream=[{"text": "one"}, {"text": "two"}],
    )

    assert stats.tokens_written == 3
    assert stats.documents_used == 2
    assert np.fromfile(tmp_path / "shard_00000.bin", dtype=np.uint16).tolist() == [3, 99, 3]


def test_prepare_streaming_dataset_closes_generator_after_token_budget(tmp_path) -> None:
    state = {"closed": False}

    def stream():
        try:
            yield {"text": "one"}
            yield {"text": "two"}
        finally:
            state["closed"] = True

    prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="unused-with-injected-stream",
        text_field="text",
        num_tokens=1,
        shard_size=10,
        tokenizer=FakeTokenizer(),
        stream=stream(),
    )

    assert state["closed"] is True


def test_fineweb_split_preparation_reserves_disjoint_documents(tmp_path) -> None:
    train_stats, val_stats = prepare_fineweb_edu_splits(
        tmp_path / "train",
        tmp_path / "val",
        train_tokens=4,
        val_tokens=2,
        shard_size=10,
        tokenizer=FakeTokenizer(),
        stream=[{"text": "one"}, {"text": "four"}, {"text": "two"}],
    )

    assert train_stats.tokens_written == 4
    assert val_stats.tokens_written == 2
    assert np.fromfile(tmp_path / "val" / "shard_00000.bin", dtype=np.uint16).tolist() == [3, 99]
    assert np.fromfile(tmp_path / "train" / "shard_00000.bin", dtype=np.uint16).tolist() == [4, 99, 3, 99]
