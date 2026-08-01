import pytest

from jarvislm import ModelConfig


def test_original_style_config() -> None:
    config = ModelConfig()

    assert config.vocab_size == 50_304
    assert config.d_model == 1_024
    assert config.n_layers == 24
    assert config.n_heads == 16
    assert config.n_kv_heads == 4
    assert config.head_dim == 64
    assert config.ffn_hidden_dim == 2_730
    assert config.uses_gqa is True


def test_tiny_config_for_local_tests() -> None:
    config = ModelConfig(
        vocab_size=1_024,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        ffn_hidden_dim=256,
    )

    assert config.head_dim == 32
    assert config.uses_gqa is False


def test_d_model_must_divide_by_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=1_000, n_heads=16)


def test_gqa_head_count_must_be_valid() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(n_heads=16, n_kv_heads=3)


def test_rope_requires_an_even_head_dimension() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        ModelConfig(d_model=120, n_heads=8)