"""Decoder-only GPT model assembled from JarvisLM model components."""

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rmsnorm import RMSNorm
from .tfblock import TransformerBlock

GPTConfig = ModelConfig


class GPT(nn.Module):
    """Decoder-only language model with optional mHC residual streams.

    RoPE is applied inside each attention module, so token embeddings are the
    only input embeddings required here.  When mHC is enabled, every block
    operates on ``[S, B, T, D]`` residual streams and a learned final readout
    returns the usual ``[B, T, D]`` representation before the LM head.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    layer_idx=layer_idx,
                    ffn_hidden_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    max_seq_len=config.max_seq_len,
                    use_flash=config.use_flash,
                    use_qk_norm=config.use_qk_norm,
                    use_diff_attn=config.use_diff_attn,
                    use_mhc=config.use_mhc,
                    n_streams=config.n_streams,
                    use_xsa=config.use_xsa,
                )
                for layer_idx in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.use_mhc:
            final_read_logits = torch.full((config.n_streams,), -2.0)
            final_read_logits[0] = 2.0
            self.final_read_logits = nn.Parameter(final_read_logits)
        else:
            self.register_parameter("final_read_logits", None)

        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight
        # Match the historical V1 initialization order.  The shared embedding
        # is visited once as an Embedding and once as the tied Linear weight.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits and, optionally, cross-entropy loss."""
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        x = self.emb_dropout(self.token_embeddings(input_ids))
        if self.config.use_mhc:
            streams = x.unsqueeze(0).expand(self.config.n_streams, -1, -1, -1)
            for block in self.blocks:
                streams = block(streams)
            read_weights = F.softmax(self.final_read_logits, dim=0).to(
                dtype=streams.dtype
            )
            x = torch.einsum("s,sbtd->btd", read_weights, streams)
        else:
            for block in self.blocks:
                x = block(x)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens using the model context window."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.max_seq_len :]
                logits, _ = self(context)
                next_token_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    k = min(top_k, next_token_logits.shape[-1])
                    threshold = torch.topk(next_token_logits, k).values[:, [-1]]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float("-inf")
                    )
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
        finally:
            self.train(was_training)
        return input_ids

    def count_parameters(self) -> dict[str, int]:
        """Return an untied-parameter-safe component breakdown."""
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        return {
            "embeddings": count(self.token_embeddings),
            "blocks": sum(count(block) for block in self.blocks),
            "final_norm": count(self.norm),
            "lm_head": 0 if self.config.tie_weights else count(self.lm_head),
            "total": count(self),
        }


__all__ = ["GPT", "GPTConfig"]
