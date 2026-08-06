import pytest
import torch

from jarvislm.model.attention import DifferentialAttention, MultiHeadAttention


@pytest.mark.parametrize("use_flash", [False, True])
def test_multi_head_attention_preserves_shape_dtype_and_gradients(
    use_flash: bool,
) -> None:
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        use_flash=use_flash,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output = attention(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert x.grad is not None
    assert attention.W_q.weight.grad is not None
    assert attention.W_k.weight.grad is not None
    assert attention.W_v.weight.grad is not None
    assert attention.W_o.weight.grad is not None
    assert attention.qk_scale.grad is not None


@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_multi_head_flash_matches_manual_causal_attention(use_qk_norm: bool) -> None:
    torch.manual_seed(0)
    manual = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        use_flash=False,
        use_qk_norm=use_qk_norm,
    ).eval()
    flash = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        use_flash=True,
        use_qk_norm=use_qk_norm,
    ).eval()
    flash.load_state_dict(manual.state_dict())
    x = torch.randn(2, 5, 16)

    torch.testing.assert_close(flash(x), manual(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_flash", [False, True])
def test_multi_head_attention_is_causal(use_flash: bool) -> None:
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=8,
        use_flash=use_flash,
    ).eval()
    prefix = torch.randn(1, 3, 16)
    future = torch.randn(1, 2, 16)

    prefix_output = attention(prefix)
    full_output = attention(torch.cat([prefix, future], dim=1))

    torch.testing.assert_close(full_output[:, :3], prefix_output)


def test_multi_head_grouped_query_attention_uses_fewer_kv_heads() -> None:
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        use_flash=False,
    )

    output = attention(torch.randn(2, 4, 16))

    assert attention.d_k == 4
    assert attention.n_rep == 2
    assert attention.W_k.out_features == 8
    assert attention.W_v.out_features == 8
    assert output.shape == (2, 4, 16)


@pytest.mark.parametrize("use_flash", [False, True])
def test_multi_head_xsa_removes_single_token_self_attention_component(
    use_flash: bool,
) -> None:
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=8,
        use_flash=use_flash,
        use_xsa=True,
    ).eval()

    output = attention(torch.randn(2, 1, 16))

    # A single causal token attends only to itself, so XSA removes all of V.
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=1e-5)


def test_multi_head_attention_rejects_incompatible_head_counts() -> None:
    with pytest.raises(AssertionError, match="divisible by n_heads"):
        MultiHeadAttention(d_model=10, n_heads=4)

    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=16, n_heads=4, n_kv_heads=3)


@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_differential_attention_preserves_shape_dtype_and_gradients(
    use_qk_norm: bool,
) -> None:
    attention = DifferentialAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        use_qk_norm=use_qk_norm,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output = attention(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert x.grad is not None
    assert attention.W_q.weight.grad is not None
    assert attention.W_k.weight.grad is not None
    assert attention.W_v.weight.grad is not None
    assert attention.W_o.weight.grad is not None
    assert attention.lambda_q1.grad is not None
    assert attention.lambda_k1.grad is not None
    assert attention.lambda_q2.grad is not None
    assert attention.lambda_k2.grad is not None
    if use_qk_norm:
        assert attention.qk_scale1.grad is not None
        assert attention.qk_scale2.grad is not None


def test_differential_attention_is_causal() -> None:
    attention = DifferentialAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
    ).eval()
    prefix = torch.randn(1, 3, 16)
    future = torch.randn(1, 2, 16)

    prefix_output = attention(prefix)
    full_output = attention(torch.cat([prefix, future], dim=1))

    torch.testing.assert_close(full_output[:, :3], prefix_output)


def test_differential_attention_lambda_matches_its_initial_value_when_zeroed() -> None:
    attention = DifferentialAttention(d_model=16, n_heads=4, layer_idx=3)

    with torch.no_grad():
        attention.lambda_q1.zero_()
        attention.lambda_k1.zero_()
        attention.lambda_q2.zero_()
        attention.lambda_k2.zero_()

    torch.testing.assert_close(attention.compute_lambda(), attention.lambda_init)


def test_differential_attention_grouped_query_attention_uses_fewer_kv_heads() -> None:
    attention = DifferentialAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
    )

    output = attention(torch.randn(2, 4, 16))

    assert attention.d_k == 4
    assert attention.d_k_half == 2
    assert attention.n_rep == 2
    assert attention.W_k.out_features == 8
    assert attention.W_v.out_features == 8
    assert output.shape == (2, 4, 16)


def test_differential_attention_rejects_invalid_head_dimensions() -> None:
    with pytest.raises(AssertionError, match="divisible by n_heads"):
        DifferentialAttention(d_model=10, n_heads=4)

    with pytest.raises(AssertionError, match="divisible by 4"):
        DifferentialAttention(d_model=24, n_heads=4)

    with pytest.raises(AssertionError):
        DifferentialAttention(d_model=16, n_heads=4, n_kv_heads=3)
