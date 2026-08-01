import torch
import torch.nn as nn
import torch.nn.functional as F



class SwiGLU(nn.Module):
    def __init__(self,d_model:int, hidden_dim:int , bias:bool = False):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        
