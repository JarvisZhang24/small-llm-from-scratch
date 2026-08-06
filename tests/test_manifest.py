import json

import pytest

from jarvislm.data import (
    TokenShardWriter,
    build_fineweb_edu_manifest,
    verify_fineweb_edu_manifest,
    write_manifest,
)


def write_tokens(directory, tokens: list[int]) -> None:
    writer = TokenShardWriter(directory, shard_size=3)
    writer.add_tokens(tokens)
    writer.flush()


def test_manifest_round_trip_verifies_complete_persistent_splits(tmp_path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    write_tokens(train_dir, [1, 2, 3, 4])
    write_tokens(val_dir, [5, 6])
    manifest_path = tmp_path / "manifest.json"

    manifest = build_fineweb_edu_manifest(
        train_dir, val_dir, expected_train_tokens=4, expected_val_tokens=2
    )
    write_manifest(manifest_path, manifest)

    verified = verify_fineweb_edu_manifest(
        manifest_path, train_dir, val_dir, expected_train_tokens=4, expected_val_tokens=2
    )
    assert verified["splits"]["train"]["shards"] == [
        "shard_00000.bin",
        "shard_00001.bin",
    ]
    assert json.loads(manifest_path.read_text()) == manifest


def test_manifest_rejects_incomplete_or_changed_shards(tmp_path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    write_tokens(train_dir, [1, 2, 3])
    write_tokens(val_dir, [4, 5])

    with pytest.raises(ValueError, match="contain 3 tokens"):
        build_fineweb_edu_manifest(
            train_dir, val_dir, expected_train_tokens=4, expected_val_tokens=2
        )

    manifest_path = tmp_path / "manifest.json"
    manifest = build_fineweb_edu_manifest(
        train_dir, val_dir, expected_train_tokens=3, expected_val_tokens=2
    )
    write_manifest(manifest_path, manifest)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="expected 4"):
        verify_fineweb_edu_manifest(
            manifest_path, train_dir, val_dir, expected_train_tokens=4, expected_val_tokens=2
        )
