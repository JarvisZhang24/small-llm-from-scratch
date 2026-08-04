import pytest
import torch

from jarvislm.model.rope import RoPECache, apply_rope, compute_rope_frequencies


def test_compute_rope_frequencies_matches_expected_angles() -> None:
    cos, sin = compute_rope_frequencies(head_dim=4, max_seq_len=3, base=100.0)

    positions = torch.arange(3, dtype=torch.float32)
    thetas = torch.tensor([1.0, 0.1])
    expected_angles = torch.outer(positions, thetas)

    assert cos.shape == (3, 2)
    assert sin.shape == (3, 2)
    torch.testing.assert_close(cos, torch.cos(expected_angles))
    torch.testing.assert_close(sin, torch.sin(expected_angles))


def test_apply_rope_matches_manual_pairwise_rotation() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]])
    freqs = compute_rope_frequencies(head_dim=4, max_seq_len=2, base=100.0)

    output = apply_rope(x, freqs)

    cos, sin = freqs
    expected = torch.empty_like(x)
    expected[..., 0::2] = x[..., 0::2] * cos[None, :, None] - x[..., 1::2] * sin[
        None, :, None
    ]
    expected[..., 1::2] = x[..., 0::2] * sin[None, :, None] + x[..., 1::2] * cos[
        None, :, None
    ]

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    torch.testing.assert_close(output, expected)


def test_apply_rope_preserves_vector_norm() -> None:
    x = torch.randn(2, 3, 4, 8, dtype=torch.float64)
    freqs = compute_rope_frequencies(head_dim=8, max_seq_len=3)

    output = apply_rope(x, freqs)

    torch.testing.assert_close(output.norm(dim=-1), x.norm(dim=-1))


def test_rope_cache_returns_prefix_and_expands_for_longer_sequences() -> None:
    cache = RoPECache(d_k=4, max_seq_len=2, base=100.0)
    initial_cos = cache.cos_f.clone()
    initial_sin = cache.sin_f.clone()

    cos, sin = cache.get_freqs(seq_len=4)

    assert cache.max_seq_len == 4
    assert cos.shape == (4, 2)
    assert sin.shape == (4, 2)
    torch.testing.assert_close(cos[:2], initial_cos)
    torch.testing.assert_close(sin[:2], initial_sin)


def test_rope_cache_returns_absolute_position_offsets() -> None:
    cache = RoPECache(d_k=4, max_seq_len=6, base=100.0)

    cos, sin = cache.get_freqs_offset(start=2, seq_length=3)

    torch.testing.assert_close(cos, cache.cos_f[2:5])
    torch.testing.assert_close(sin, cache.sin_f[2:5])

    with pytest.raises(AssertionError, match="exceeds RoPE table"):
        cache.get_freqs_offset(start=4, seq_length=3)
