import pytest
import torch

from jarvislm import RMSNorm


def test_rmsnorm_preserves_shape_and_dtype() -> None:
    norm = RMSNorm(d_model=8)
    x = torch.randn(2, 4, 8)

    output = norm(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_rmsnorm_matches_manual_formula() -> None:
    eps = 1e-6
    norm = RMSNorm(d_model=3, eps=eps)

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5]))

    x = torch.tensor(
        [
            [[3.0, 4.0, 0.0]],
            [[1.0, 2.0, 2.0]],
        ]
    )

    output = norm(x)

    inverse_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = x * inverse_rms * norm.weight

    torch.testing.assert_close(output, expected)


def test_rmsnorm_has_gradients() -> None:
    norm = RMSNorm(d_model=8)
    x = torch.randn(2, 4, 8, requires_grad=True)

    loss = norm(x).sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None


def test_rmsnorm_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="d_model"):
        RMSNorm(d_model=0)

    with pytest.raises(ValueError, match="eps"):
        RMSNorm(d_model=8, eps=0.0)