import math

import torch
import torch.nn.functional as F
from torch import nn

from .rmsnorm import RMSNorm
from .rope import RoPECache, apply_rope


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with optional GQA, RoPE, QK-Norm, and Flash Attention support.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_flash: bool = True,
        use_qk_norm: bool = True,
        use_xsa: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads  # dimension per head
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.use_qk_norm = use_qk_norm
        self.use_xsa = use_xsa

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        assert n_heads % self.n_kv_heads == 0  # Check if number of KV heads makes sense
        # If using GQA
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE cache
        self.rope_cache = RoPECache(self.d_k, max_seq_len)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # QK Norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(torch.ones(n_heads) * (self.d_k**0.5))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask (not used if using F.scaled_dot_product_attention with is_causal)

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1 - project to Q, K, V
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)  # [batch, seq_len, d_model]
        v = self.W_v(x)  # [batch, seq_len, d_model]

        # Step 2 - Split into multiple heads
        q = q.view(
            batch_size, seq_len, self.n_heads, self.d_k
        )  # [batch, seq_len, n_heads, d_k]
        k = k.view(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [batch, seq_len, n_kv_heads, d_k]
        v = v.view(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [batch, seq_len, n_kv_heads, d_k]

        # Step 2.b - Apply QK-Norm if present
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
            k = F.normalize(k, dim=-1)

        # Step 3 - Apply RoPE to q and k
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Step 3.b - if GQA, expand K and V. Apply on the 3rd dim
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Step 4 - Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        k = k.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        v = v.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]

        # Step 5 - Compute attention using Flash Attention
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # Automatically applies causal mask,
                scale=1.0 if self.use_qk_norm else None,
            )
        else:
            # Manual attention computation (for learning)
            scale = 1.0 if self.use_qk_norm else (self.d_k**-0.5)
            attn_scores = (
                torch.matmul(q, k.transpose(-2, -1)) * scale
            )  # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len] -> [batch, n_heads, seq_len, seq_len]

            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
            attn_scores = (
                attn_scores + causal_mask
            )  # Broadcasting will apply mask to all batches and heads
            attn_weights = F.softmax(
                attn_scores, dim=-1
            )  # [batch, n_heads, seq_len, seq_len]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(
                attn_weights, v
            )  # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_k] - > [batch, n_heads, seq_len, d_k]

        # Step 5.b — Exclusive Self-Attention (XSA): remove the component of the
        # attention output aligned with each token's own value vector.
        # attn_output and v are both [batch, n_heads, seq_len, d_k] here.
        # v was already GQA-expanded and transposed above, so shapes align.
        if self.use_xsa:
            # projection of attn_output onto v, per (batch, head, position)
            dot = (attn_output * v).sum(
                dim=-1, keepdim=True
            )  # [batch, n_heads, seq_len, 1]
            denom = (
                v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            )  # [bath, n_heads, seq_len, 1]
            projection = (dot / denom) * v  # [batch, n_heads, seq_len, d_k]
            attn_output = attn_output - projection

        # Step 6 - Reshape back
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, n_heads, d_k]
        attn_output = attn_output.contiguous().view(
            batch_size, seq_len, d_model
        )  # [batch, seq_len, d_model]

        # Step 7 - Apply output projection
        output = self.W_o(attn_output)

        return output


class DifferentialAttention(nn.Module):
    """
    Differential Attention module with optional GQA, RoPE, QK-Norm, and Flash Attention support.
    This module computes two attention outputs and combines them using a learnable lambda parameter.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        layer_idx: int = 0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        self.n_rep = n_heads // self.n_kv_heads
        assert n_heads % self.n_kv_heads == 0

        self.d_model = d_model
        self.d_k = d_model // n_heads
        assert self.d_k % 4 == 0, (
            "DifferentialAttention + RoPE requires head_dim divisible by 4"
        )
        self.d_k_half = self.d_k // 2
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Lambda init (unchanged from before)
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.register_buffer(
            "lambda_init", torch.tensor(lambda_init, dtype=torch.float32)
        )
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)

        self.head_norm = RMSNorm(self.d_k)
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)

        if use_qk_norm:
            # Each half has dimension d_k_half, so scale init uses sqrt(d_k_half)
            self.qk_scale1 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half**0.5))
            self.qk_scale2 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half**0.5))

    def compute_lambda(self) -> torch.Tensor:
        """
        Compute the current differential attention lambda.

        Returns:
            Scalar tensor.
        """
        return (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)  # [batch, seq_len, d_model]
        v = self.W_v(x)  # [batch, seq_len, d_model]

        # Change dims
        q = q.view(
            batch_size, seq_len, self.n_heads, self.d_k
        )  # [batch, seq_len, n_heads, d_k]
        k = k.view(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [batch, seq_len, n_kv_heads, d_k]
        v = v.view(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [batch, seq_len, n_kv_heads, d_k]

        # Split into two halves (q1/q2 each [batch, seq, n_heads, d_k_half])
        # (k1/k2 each [batch, seq, n_kv_heads, d_k_half])
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # If using QK-norm, normalize:
        if self.use_qk_norm:
            q1 = F.normalize(q1, dim=-1) * self.qk_scale1.view(1, 1, -1, 1)
            q2 = F.normalize(q2, dim=-1) * self.qk_scale2.view(1, 1, -1, 1)
            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)

        # RoPE on each half
        freqs = self.rope_cache.get_freqs(seq_len)
        q1 = apply_rope(q1, freqs)
        q2 = apply_rope(q2, freqs)
        k1 = apply_rope(k1, freqs)
        k2 = apply_rope(k2, freqs)

        # GQA expansion — repeat k1, k2, v from n_kv_heads to n_heads
        if self.n_rep > 1:
            k1 = k1.repeat_interleave(self.n_rep, dim=2)
            k2 = k2.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose all to [batch, n_heads, seq, dim]
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)

        # Lambda
        lam = self.compute_lambda().to(dtype=v.dtype)

        # Two attention outputs using Flash Attention for speedup
        attn1_output = F.scaled_dot_product_attention(
            q1,
            k1,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # Automatically applies causal mask,
            scale=1.0 if self.use_qk_norm else (self.d_k_half**-0.5),
        )

        attn2_output = F.scaled_dot_product_attention(
            q2,
            k2,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # Automatically applies causal mask,
            scale=1.0 if self.use_qk_norm else (self.d_k_half**-0.5),
        )
        # Apply to V, reshape, RMSNorm, output
        attn_output = (
            attn1_output - lam * attn2_output
        )  # Subtract the two attention outputs, equivalent to (A1 - λA2) @ V
        attn_output = attn_output.transpose(1, 2)  # [batch, seq, n_heads, d_k]
        attn_output = self.head_norm(
            attn_output
        )  # RMSNorm over last dim d_k, per token per head
        scale_factor = (1 - self.lambda_init).to(dtype=attn_output.dtype)
        attn_output = attn_output * scale_factor
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)
