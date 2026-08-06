"""Multi-stream Hyper-Connection-style residual routing.

``MHCResidual`` wraps one Transformer sublayer (attention or MLP).  Instead of
maintaining one residual tensor, it maintains several residual streams with
shape ``[n_streams, batch, sequence, d_model]``.  A Sinkhorn-normalized mixing
matrix exchanges information between streams, then learned read and write
weights connect the streams to the wrapped sublayer.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def sinkhorn(log_weights: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Return a doubly stochastic matrix from square, unconstrained logits.

    Normalization is performed in float32 so the operation remains stable when
    a model is trained in float16 or bfloat16.  The result is restored to the
    input dtype before returning.
    """
    if log_weights.ndim < 2:
        raise ValueError("log_weights must have at least two dimensions")
    if log_weights.shape[-1] != log_weights.shape[-2]:
        raise ValueError("log_weights must be square in its last two dimensions")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    original_dtype = log_weights.dtype
    log_probabilities = log_weights.float()
    for _ in range(n_iters):
        log_probabilities = F.log_softmax(log_probabilities, dim=-1)
        log_probabilities = F.log_softmax(log_probabilities, dim=-2)
    return log_probabilities.exp().to(dtype=original_dtype)


class MHCResidual(nn.Module):
    """Route a Transformer sublayer through multiple learned residual streams.

    ``forward`` receives the current residual streams and callable sublayer
    components.  This keeps attention and MLP implementations independent of
    mHC while allowing a Transformer block to wrap either of them.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        identity_bias: float = 3.0,
        sinkhorn_iters: int = 5,
        write_init: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")

        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        self.log_mixing = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.register_buffer("identity_bias", torch.eye(n_streams) * identity_bias)

        read_logits = torch.zeros(n_streams)
        read_logits[0] = 1.0
        self.read_logits = nn.Parameter(read_logits)

        write_gates = torch.full((n_streams,), 0.25 * write_init)
        write_gates[0] = write_init
        self.write_gates = nn.Parameter(write_gates)

    def mixing_matrix(self) -> torch.Tensor:
        """Return the learned stream-mixing matrix constrained by Sinkhorn."""
        return sinkhorn(
            self.log_mixing + self.identity_bias,
            n_iters=self.sinkhorn_iters,
        )

    def forward(
        self,
        streams: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
        norm: nn.Module,
    ) -> torch.Tensor:
        """Mix streams, apply one pre-norm sublayer, and write its update back.

        Args:
            streams: Residual streams with shape ``[S, B, T, D]``.
            sublayer: Attention or MLP callable mapping ``[B, T, D]`` to the
                same shape.
            norm: Pre-normalization module applied before ``sublayer``.
        """
        if streams.ndim != 4:
            raise ValueError(
                f"streams must have shape [S, B, T, D], got {tuple(streams.shape)}"
            )
        n_streams, _, _, d_model = streams.shape
        if n_streams != self.n_streams:
            raise ValueError(f"expected {self.n_streams} streams, got {n_streams}")
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        mixing = self.mixing_matrix().to(dtype=streams.dtype)
        mixed_streams = torch.einsum("ij,jbtd->ibtd", mixing, streams)

        read_weights = F.softmax(self.read_logits, dim=0).to(dtype=streams.dtype)
        sublayer_input = torch.einsum("s,sbtd->btd", read_weights, mixed_streams)
        update = sublayer(norm(sublayer_input))
        if update.shape != sublayer_input.shape:
            raise ValueError(
                "sublayer must preserve [batch, sequence, d_model], "
                f"got {tuple(update.shape)}"
            )

        write_gates = self.write_gates.view(n_streams, 1, 1, 1).to(
            dtype=update.dtype
        )
        return mixed_streams + write_gates * update.unsqueeze(0)
