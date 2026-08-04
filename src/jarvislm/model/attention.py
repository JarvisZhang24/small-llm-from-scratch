import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .rope import RoPECache, apply_rope


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE positional encoding
    """
