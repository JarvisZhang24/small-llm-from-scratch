import pytest
import torch

from jarvislm.model.attention import DifferentialAttention, MultiHeadAttention
from jarvislm.model.tfblock import TransformerBlock


def test_standard_transformer_block_preserves_shape_dtype_and_gradients() -> None:
    block = TransformerBlock(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=32,
        max_seq_len=8,
        use_diff_attn=False,
        use_flash=False,
        use_mhc=False,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output = block(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert x.grad is not None
    assert block.norm1.weight.grad is not None
    assert block.norm2.weight.grad is not None
    assert block.attention.W_q.weight.grad is not None
    assert block.mlp.w_gate.weight.grad is not None


def test_transformer_block_uses_config_compatible_default_ffn_dimension() -> None:
    block = TransformerBlock(
        d_model=24,
        n_heads=3,
        use_diff_attn=False,
        use_flash=False,
    )

    assert block.mlp.hidden_dim == int(8 / 3 * 24)


@pytest.mark.parametrize(
    ("use_diff_attn", "use_xsa", "expected_attention_type"),
    [
        (True, False, DifferentialAttention),
        (False, False, MultiHeadAttention),
        (False, True, MultiHeadAttention),
    ],
)
def test_transformer_block_selects_requested_attention_implementation(
    use_diff_attn: bool, use_xsa: bool, expected_attention_type: type[torch.nn.Module]
) -> None:
    block = TransformerBlock(
        d_model=16,
        n_heads=4,
        ffn_hidden_dim=32,
        max_seq_len=8,
        use_diff_attn=use_diff_attn,
        use_xsa=use_xsa,
    )

    assert isinstance(block.attention, expected_attention_type)
    if not use_diff_attn:
        assert block.attention.use_xsa is use_xsa


def test_mhc_transformer_block_preserves_stream_shape_and_gradients() -> None:
    block = TransformerBlock(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=32,
        max_seq_len=8,
        use_diff_attn=True,
        use_mhc=True,
        n_streams=2,
    )
    streams = torch.randn(2, 2, 5, 16, requires_grad=True)

    output = block(streams)
    output.square().mean().backward()

    assert output.shape == streams.shape
    assert output.dtype == streams.dtype
    assert streams.grad is not None
    assert block.mhc_attn.log_mixing.grad is not None
    assert block.mhc_mlp.log_mixing.grad is not None
    assert block.attention.W_q.weight.grad is not None
    assert block.mlp.w_down.weight.grad is not None


def test_transformer_block_enables_mhc_by_default() -> None:
    block = TransformerBlock(
        d_model=16,
        n_heads=4,
        ffn_hidden_dim=32,
        max_seq_len=8,
    )
    streams = torch.randn(4, 1, 3, 16)

    output = block(streams)

    assert block.use_mhc is True
    assert output.shape == streams.shape


def test_transformer_block_rejects_invalid_configuration_and_input_shapes() -> None:
    with pytest.raises(ValueError, match="d_model"):
        TransformerBlock(d_model=0, n_heads=1)
    with pytest.raises(ValueError, match="ffn_hidden_dim"):
        TransformerBlock(d_model=16, n_heads=4, ffn_hidden_dim=0)
    with pytest.raises(ValueError, match="n_streams"):
        TransformerBlock(d_model=16, n_heads=4, use_mhc=True, n_streams=0)

    standard_block = TransformerBlock(
        d_model=16,
        n_heads=4,
        ffn_hidden_dim=32,
        use_diff_attn=False,
        use_mhc=False,
    )
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        standard_block(torch.randn(2, 3, 4, 16))
    with pytest.raises(ValueError, match="expected d_model=16"):
        standard_block(torch.randn(2, 3, 8))

    mhc_block = TransformerBlock(
        d_model=16,
        n_heads=4,
        ffn_hidden_dim=32,
        use_mhc=True,
        n_streams=2,
    )
    with pytest.raises(ValueError, match=r"\[S, B, T, D\]"):
        mhc_block(torch.randn(2, 3, 16))
    with pytest.raises(ValueError, match="expected 2 streams"):
        mhc_block(torch.randn(3, 1, 2, 16))
