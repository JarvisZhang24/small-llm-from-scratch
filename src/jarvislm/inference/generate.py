"""Inference utilities for checkpoints produced by :mod:`jarvislm.training`.

This module deliberately lives outside ``jarvislm.model``.  The model package
defines the network; inference additionally needs checkpoint format knowledge,
tokenization, and sampling rules.  In particular, JarvisLM's embedding table is
padded from the GPT-2 vocabulary of 50,257 IDs to 50,304 IDs.  Padding IDs must
never be sampled as text tokens.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from jarvislm.model import GPT, ModelConfig


class TextTokenizer(Protocol):
    """Minimal tokenizer interface needed by :func:`generate_text`."""

    vocab_size: int
    eos_id: int

    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


def _clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the prefix added when a training model was ``torch.compile``d."""
    return {
        name.removeprefix("_orig_mod."): tensor for name, tensor in state_dict.items()
    }


def load_inference_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    use_ema: bool = True,
) -> GPT:
    """Load a JarvisLM training checkpoint for evaluation or generation.

    ``use_ema=True`` selects the EMA copy when it is available.  A checkpoint
    without EMA weights is still usable and automatically falls back to its raw
    model weights, which makes checkpoints from EMA-disabled smoke tests useful.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping payload")

    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise TypeError("checkpoint is missing its 'model_config' mapping")
    config = ModelConfig(**dict(raw_config))

    state_name = "ema" if use_ema and "ema" in checkpoint else "model"
    state_dict = checkpoint.get(state_name)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"checkpoint is missing its '{state_name}' state dict")

    model = GPT(config)
    model.load_state_dict(_clean_state_dict(state_dict))
    return model.to(device).eval()


def _validate_sampling_args(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    valid_vocab_size: int,
    model_vocab_size: int,
    eos_id: int | None,
) -> None:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have non-empty shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0 < valid_vocab_size <= model_vocab_size:
        raise ValueError("valid_vocab_size must be in the model vocabulary range")
    if eos_id is not None and not 0 <= eos_id < valid_vocab_size:
        raise ValueError("eos_id must be inside the valid vocabulary")


@torch.no_grad()
def generate_token_ids(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    valid_vocab_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively sample IDs while excluding padded vocabulary entries.

    ``valid_vocab_size`` is normally ``GPT2Tokenizer.vocab_size`` (50,257),
    rather than the model's padded 50,304 output dimension.
    """
    _validate_sampling_args(
        input_ids,
        max_new_tokens,
        temperature,
        top_k,
        valid_vocab_size,
        model.config.vocab_size,
        eos_id,
    )
    if input_ids.min().item() < 0 or input_ids.max().item() >= valid_vocab_size:
        raise ValueError("input_ids must contain only valid tokenizer IDs")

    was_training = model.training
    model.eval()
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    try:
        for _ in range(max_new_tokens):
            context = input_ids[:, -model.config.max_seq_len :]
            logits, _ = model(context)
            next_logits = logits[:, -1, :].float() / temperature
            next_logits[:, valid_vocab_size:] = float("-inf")

            if top_k is not None:
                k = min(top_k, valid_vocab_size)
                threshold = torch.topk(next_logits, k).values[:, [-1]]
                next_logits.masked_fill_(next_logits < threshold, float("-inf"))

            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            if eos_id is not None and finished.any():
                next_token[finished] = eos_id
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_id is not None:
                finished |= next_token.squeeze(1).eq(eos_id)
                if finished.all():
                    break
    finally:
        model.train(was_training)
    return input_ids


def generate_text(
    model: GPT,
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
) -> str:
    """Generate one continuation from ``prompt`` and decode it to text."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    token_ids = tokenizer.encode(prompt)
    # GPT-2 encodes an empty string to no IDs; EOT provides a valid initial
    # context for that otherwise useful case.
    if not token_ids:
        token_ids = [tokenizer.eos_id]
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated_ids = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        valid_vocab_size=tokenizer.vocab_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(generated_ids[0].tolist())


__all__ = ["generate_text", "generate_token_ids", "load_inference_model"]
