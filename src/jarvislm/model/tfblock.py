"""Pre-norm decoder Transformer block with optional mHC residual streams."""

import torch
from torch import nn

from .attention import DifferentialAttention, MultiHeadAttention
from .mhc import MHCResidual
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """A decoder block with attention and SwiGLU sublayers.

    In normal mode, inputs and outputs have shape ``[B, T, D]`` and the block
    uses ordinary residual additions.  With ``use_mhc=True``, the caller owns
    the residual streams: inputs and outputs have shape ``[S, B, T, D]``.
    Each sublayer is then routed through its own :class:`MHCResidual` wrapper.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        layer_idx: int = 0,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_flash: bool = True,
        use_qk_norm: bool = True,
        use_diff_attn: bool = True,
        use_mhc: bool = True,
        n_streams: int = 4,
        use_xsa: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if ffn_hidden_dim is None:
            ffn_hidden_dim = int(8 / 3 * d_model)
        if ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if use_mhc and n_streams <= 0:
            raise ValueError("n_streams must be positive when use_mhc=True")

        self.d_model = d_model
        self.use_mhc = use_mhc
        self.n_streams = n_streams

        self.norm1 = RMSNorm(d_model)
        if use_diff_attn:
            self.attention: nn.Module = DifferentialAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                layer_idx=layer_idx,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_qk_norm=use_qk_norm,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_flash=use_flash,
                use_qk_norm=use_qk_norm,
                use_xsa=use_xsa,
            )

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model=d_model, hidden_dim=ffn_hidden_dim, bias=False)

        if use_mhc:
            self.mhc_attn = MHCResidual(d_model=d_model, n_streams=n_streams)
            self.mhc_mlp = MHCResidual(d_model=d_model, n_streams=n_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with standard or multi-stream residuals."""
        if not self.use_mhc:
            if x.ndim != 3:
                raise ValueError(
                    "normal TransformerBlock input must have shape [B, T, D], "
                    f"got {tuple(x.shape)}"
                )
            if x.shape[-1] != self.d_model:
                raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

            x = x + self.attention(self.norm1(x))
            return x + self.mlp(self.norm2(x))

        if x.ndim != 4:
            raise ValueError(
                "mHC TransformerBlock input must have shape [S, B, T, D], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] != self.n_streams:
            raise ValueError(f"expected {self.n_streams} streams, got {x.shape[0]}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        streams = self.mhc_attn(x, self.attention, self.norm1)
        return self.mhc_mlp(streams, self.mlp, self.norm2)
