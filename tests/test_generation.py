from dataclasses import asdict

import pytest
import torch
from torch import nn

from jarvislm.inference import generate_text, generate_token_ids, load_inference_model
from jarvislm.model import GPT, ModelConfig


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        max_seq_len=4,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        ffn_hidden_dim=16,
        use_flash=False,
        use_qk_norm=False,
        use_diff_attn=False,
        use_mhc=False,
    )


class FakeTokenizer:
    vocab_size = 7
    eos_id = 6

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        token_ids = [1] if text else []
        return token_ids + ([self.eos_id] if add_eos else [])

    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class PaddedLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = tiny_config()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size), -10.0, device=input_ids.device
        )
        logits[..., 1] = 2.0
        logits[..., 7] = 100.0  # padded ID: this must never be emitted.
        return logits, None


def test_load_inference_model_uses_ema_and_compiled_state_keys(tmp_path) -> None:
    config = tiny_config()
    model = GPT(config)
    raw_state = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
    ema_state = {
        f"_orig_mod.{name}": torch.ones_like(value)
        for name, value in model.state_dict().items()
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model_config": asdict(config), "model": raw_state, "ema": ema_state},
        checkpoint_path,
    )

    loaded_ema = load_inference_model(checkpoint_path, use_ema=True)
    loaded_raw = load_inference_model(checkpoint_path, use_ema=False)

    assert torch.equal(loaded_ema.token_embeddings.weight, torch.ones_like(model.token_embeddings.weight))
    assert torch.equal(loaded_raw.token_embeddings.weight, torch.zeros_like(model.token_embeddings.weight))
    assert loaded_ema.training is False


def test_generation_masks_padded_vocabulary_and_restores_training_mode() -> None:
    model = PaddedLogitModel().train()
    generated = generate_token_ids(
        model,  # type: ignore[arg-type]
        torch.tensor([[1]]),
        max_new_tokens=2,
        valid_vocab_size=7,
        top_k=1,
    )

    assert generated.tolist() == [[1, 1, 1]]
    assert model.training is True


def test_generate_text_handles_empty_prompt_and_validation() -> None:
    tokenizer = FakeTokenizer()
    model = PaddedLogitModel()

    assert generate_text(model, tokenizer, "", max_new_tokens=1, top_k=1) == "6,1"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid_vocab_size"):
        generate_token_ids(
            model,  # type: ignore[arg-type]
            torch.tensor([[1]]),
            max_new_tokens=1,
            valid_vocab_size=9,
        )
