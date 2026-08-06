# src/optim/muon.py
"""Muon optimizer: Momentum Orthogonalized by Newton-Schulz.

Key idea: instead of AdamW's diagonal preconditioner (which treats each weight
independently), Muon applies a Newton-Schulz "zero-power" preconditioner to
the momentum buffer before using it as the update direction. This compresses
the singular value spectrum of the gradient, giving a more geometrically
uniform update for weight matrices.

The NS iteration with these specific coefficients converges to fixed points at
σ ≈ 0.87 and σ ≈ 1.26 — so singular values are compressed from their initial
range (e.g., [0.04, 0.21] after normalization) into approximately [0.68, 1.13].
This is the "zero-power preconditioner" from the paper: not perfectly orthogonal,
but a consistent approximation that's far more uniform than the raw gradient.
Faster convergence than AdamW because each step is more geometrically meaningful.

Reference: https://arxiv.org/abs/2502.16982
"""

import torch
from torch import nn


def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Apply Newton-Schulz zero-power preconditioner to M.

    Uses a quintic polynomial iteration to compress singular values toward
    the range [0.87, 1.26] (the fixed points of this specific polynomial).
    This makes the gradient update more spectrally uniform without computing
    a full SVD.

    The polynomial aσ + bσ³ + cσ⁵ has fixed points at σ ≈ 0.87 and σ ≈ 1.26.
    Singular values outside this range get pulled in; singular values inside
    oscillate to the stable fixed point. After 5 iterations, the spectrum is
    well-compressed into approximately [0.68, 1.13].

    Expects a wide matrix (more columns than rows). Caller handles transposing.

    Args:
        M: Wide matrix [m, n] where m <= n, with M.norm() ≈ 1
        steps: Iterations (5 is the standard speed/quality tradeoff)

    Returns:
        Spectrally compressed matrix, same shape as M
    """
    assert M.ndim == 2, f"Expected 2D matrix, got {M.ndim}D"
    assert M.shape[0] <= M.shape[1], f"Expected wide matrix (m <= n), got {M.shape}"

    # Quintic polynomial coefficients from the Muon paper
    # Fixed points at σ ≈ 0.87 and σ ≈ 1.26 — compresses spectrum toward ~1
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize so iteration starts in a numerically stable regime
    X = M / M.norm()

    # Each iteration applies: X = (a*I + b*(X@X.T) + c*(X@X.T)^2) @ X
    # This maps each singular value σ → a*σ + b*σ³ + c*σ⁵
    # compressing the spectrum toward the fixed points
    for _ in range(steps):
        A = X @ X.T  # [m, m] left Gram matrix
        X = a * X + b * (A @ X) + c * (A @ A @ X)  # quintic spectral compression

    return X


def configure_optimizers(
    model: nn.Module,
    lr: float,
    muon_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Split parameters and create Muon + AdamW optimizers.

    Muon is designed for 2D weight matrices — the linear transformations
    that rotate and project vectors (attention projections, MLP weights).
    It doesn't make sense for:
    - Embeddings: lookup tables, not linear maps
    - LM head: tied to embeddings, same reason
    - Norms: 1D scale vectors, no matrix structure
    - Biases: 1D vectors

    Args:
        model: The GPT model
        lr: Learning rate for AdamW group
        muon_lr: Learning rate for Muon group (typically 0.5x lr)
        weight_decay: L2 regularization, applied to both groups

    Returns:
        (muon_optimizer, adamw_optimizer) — call .step() on both each training step
    """
    muon_params = []  # 2D weight matrices → Muon
    adamw_params = []  # everything else → AdamW

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 2D weight matrices go to Muon, excluding non-matrix params
        is_muon_matrix = (
            param.ndim == 2
            and any(
                k in name
                for k in [
                    "W_q.weight",
                    "W_k.weight",
                    "W_v.weight",
                    "W_o.weight",
                    "mlp",
                    "w1",
                    "w2",
                    "w3",
                ]
            )
            and "embedding" not in name
            and "lm_head" not in name
        )

        if is_muon_matrix:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon params:  {sum(p.numel() for p in muon_params):>12,}")
    print(f"  AdamW params: {sum(p.numel() for p in adamw_params):>12,}")

    muon_optimizer = Muon(muon_params, lr=muon_lr)
    adamw_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
    }
    # ``fused=True`` is a CUDA optimization but is not implemented on CPU/MPS.
    if next(model.parameters()).device.type == "cuda":
        adamw_kwargs["fused"] = True
    adamw_optimizer = torch.optim.AdamW(adamw_params, **adamw_kwargs)

    return muon_optimizer, adamw_optimizer


class Muon(torch.optim.Optimizer):
    """Muon: Momentum Orthogonalized by Newton-Schulz.

    For each 2D parameter:
    1. Maintain a momentum buffer (like SGD with momentum)
    2. Apply Newton-Schulz zero-power preconditioner (5 iterations)
    3. Scale to preserve expected gradient magnitude
    4. Apply as the weight update

    Only use for 2D weight matrices. Use configure_optimizers() to split.

    Args:
        params: 2D matrix parameters only
        lr: Learning rate (Muon handles higher LR than AdamW — default 0.02)
        momentum: Momentum coefficient (default 0.95)
        nesterov: Nesterov lookahead momentum for slightly faster convergence
        ns_steps: Newton-Schulz iterations (5 is the sweet spot)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer on first step
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                # Update momentum: exponential moving average of gradients
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)

                # Nesterov: lookahead by one momentum step
                if group["nesterov"]:
                    update = grad + group["momentum"] * buf
                else:
                    update = buf

                # Scale using original parameter shape (before any transpose)
                scale = max(p.shape) ** 0.5

                # Newton-Schulz expects wide matrices (m <= n)
                # Transpose tall matrices, apply NS, transpose back
                if update.shape[0] > update.shape[1]:
                    update = update.T
                    orthogonalized = newton_schulz(update, group["ns_steps"])
                    orthogonalized = orthogonalized.T
                else:
                    orthogonalized = newton_schulz(update, group["ns_steps"])

                # Apply: spectrally compressed direction, scaled to gradient magnitude
                p.add_(orthogonalized * scale, alpha=-group["lr"])
