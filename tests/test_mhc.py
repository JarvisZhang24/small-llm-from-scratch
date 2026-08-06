import pytest
import torch
from torch import nn

from jarvislm.model.mhc import MHCResidual, sinkhorn


def test_sinkhorn_returns_doubly_stochastic_matrix() -> None:
    matrix = sinkhorn(torch.randn(4, 4), n_iters=30)

    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(4), atol=1e-4, rtol=0)
    assert torch.all(matrix >= 0)


def test_sinkhorn_preserves_dtype() -> None:
    matrix = sinkhorn(torch.randn(3, 3, dtype=torch.float64), n_iters=20)

    assert matrix.dtype == torch.float64


@pytest.mark.parametrize(
    ("log_weights", "n_iters", "message"),
    [
        (torch.randn(4), 5, "at least two dimensions"),
        (torch.randn(2, 3), 5, "square"),
        (torch.randn(2, 2), 0, "positive"),
    ],
)
def test_sinkhorn_rejects_invalid_arguments(
    log_weights: torch.Tensor, n_iters: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sinkhorn(log_weights, n_iters=n_iters)


def test_mhc_residual_preserves_shape_dtype_and_gradients() -> None:
    residual = MHCResidual(d_model=8, n_streams=2)
    streams = torch.randn(2, 3, 4, 8, requires_grad=True)
    norm = nn.LayerNorm(8)
    sublayer = nn.Linear(8, 8, bias=False)

    output = residual(streams, sublayer, norm)
    output.square().mean().backward()

    assert output.shape == streams.shape
    assert output.dtype == streams.dtype
    assert streams.grad is not None
    assert residual.log_mixing.grad is not None
    assert residual.read_logits.grad is not None
    assert residual.write_gates.grad is not None
    assert sublayer.weight.grad is not None
    assert norm.weight.grad is not None


def test_mhc_mixing_matrix_is_doubly_stochastic() -> None:
    residual = MHCResidual(d_model=8, n_streams=3, sinkhorn_iters=30)
    matrix = residual.mixing_matrix()

    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(3), atol=1e-4, rtol=0)


def test_mhc_stream_specific_write_gates_break_symmetric_streams() -> None:
    residual = MHCResidual(d_model=8, n_streams=2)
    x = torch.randn(1, 4, 8)
    streams = x.unsqueeze(0).repeat(2, 1, 1, 1)
    norm = nn.Identity()
    sublayer = nn.Linear(8, 8, bias=False)

    output = residual(streams, sublayer, norm)

    assert not torch.allclose(output[0], output[1])


def test_mhc_rejects_invalid_configuration_and_stream_shapes() -> None:
    with pytest.raises(ValueError, match="d_model"):
        MHCResidual(d_model=0)
    with pytest.raises(ValueError, match="n_streams"):
        MHCResidual(d_model=8, n_streams=0)
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        MHCResidual(d_model=8, sinkhorn_iters=0)

    residual = MHCResidual(d_model=8, n_streams=2)
    norm = nn.Identity()
    sublayer = nn.Identity()
    with pytest.raises(ValueError, match=r"\[S, B, T, D\]"):
        residual(torch.randn(2, 3, 8), sublayer, norm)
    with pytest.raises(ValueError, match="expected 2 streams"):
        residual(torch.randn(3, 1, 2, 8), sublayer, norm)
    with pytest.raises(ValueError, match="expected d_model=8"):
        residual(torch.randn(2, 1, 2, 4), sublayer, norm)


def test_mhc_rejects_sublayer_that_changes_shape() -> None:
    residual = MHCResidual(d_model=8, n_streams=2)
    streams = torch.randn(2, 1, 2, 8)

    with pytest.raises(ValueError, match="sublayer must preserve"):
        residual(streams, nn.Linear(8, 4), nn.Identity())
