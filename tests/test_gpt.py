import pytest
import torch

from jarvislm import GPT, GPTConfig, ModelConfig


def make_tiny_config(**overrides: object) -> ModelConfig:
    options: dict[str, object] = {
        "vocab_size": 32,
        "max_seq_len": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 32,
        "use_flash": False,
        "use_diff_attn": True,
        "use_mhc": True,
        "n_streams": 2,
    }
    options.update(overrides)
    return ModelConfig(**options)


def test_gpt_config_aliases_model_config() -> None:
    assert GPTConfig is ModelConfig


def test_gpt_forward_with_mhc_returns_logits_loss_and_gradients() -> None:
    config = make_tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))

    logits, loss = model(input_ids, targets)
    assert loss is not None
    loss.backward()

    assert logits.shape == (2, 5, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.token_embeddings.weight is model.lm_head.weight
    assert model.token_embeddings.weight.grad is not None
    assert model.final_read_logits.grad is not None
    assert model.blocks[0].mhc_attn.log_mixing.grad is not None


def test_gpt_forward_without_mhc_uses_standard_blocks() -> None:
    config = make_tiny_config(use_mhc=False, use_diff_attn=False)
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, loss = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)
    assert loss is None
    assert model.final_read_logits is None
    assert all(not block.use_mhc for block in model.blocks)


def test_gpt_exposes_no_unsafe_generate_method() -> None:
    # Sampling must go through jarvislm.inference.generate_token_ids, which is
    # the only path that excludes the padded vocabulary entries.
    assert not hasattr(GPT(make_tiny_config()), "generate")


def test_gpt_count_parameters_excludes_tied_lm_head() -> None:
    tied_model = GPT(make_tiny_config(tie_weights=True))
    untied_model = GPT(make_tiny_config(tie_weights=False))

    tied_counts = tied_model.count_parameters()
    untied_counts = untied_model.count_parameters()

    assert tied_counts["lm_head"] == 0
    assert untied_counts["lm_head"] == 32 * 16
    assert untied_counts["total"] == tied_counts["total"] + 32 * 16


def test_gpt_rejects_invalid_input_and_target_arguments() -> None:
    config = make_tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    with pytest.raises(ValueError, match=r"\[batch, sequence\]"):
        model(torch.randint(0, config.vocab_size, (2, 3, 4)))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.randint(0, config.vocab_size, (1, 9)))
    with pytest.raises(ValueError, match="same shape"):
        model(input_ids, torch.randint(0, config.vocab_size, (2, 4)))
