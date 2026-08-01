import pytest
import torch
import torch.nn.functional as F

from jarvislm import SwiGLU


def test_swiglu_preserves_shape() -> None:
    module = SwiGLU(d_model=8, hidden_dim=16)
    x = torch.randn(2, 4, 8)

    output = module(x)

    assert output.shape == x.shape


def test_swiglu_matches_formula() -> None:
    module = SwiGLU(d_model=4, hidden_dim=8)
    x = torch.randn(2, 3, 4)

    output = module(x)

    gate = F.silu(module.w_gate(x))
    value = module.w_up(x)
    expected = module.w_down(gate * value)

    torch.testing.assert_close(output, expected)


def test_swiglu_has_gradients() -> None:
    module = SwiGLU(d_model=8, hidden_dim=16)
    x = torch.randn(2, 4, 8, requires_grad=True)

    loss = module(x).sum()
    loss.backward()

    assert x.grad is not None
    assert module.w_gate.weight.grad is not None
    assert module.w_up.weight.grad is not None
    assert module.w_down.weight.grad is not None


def test_swiglu_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="d_model"):
        SwiGLU(d_model=0, hidden_dim=16)

    with pytest.raises(ValueError, match="hidden_dim"):
        SwiGLU(d_model=8, hidden_dim=0)