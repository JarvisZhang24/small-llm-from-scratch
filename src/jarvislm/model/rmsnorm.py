import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")

        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
