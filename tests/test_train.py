import importlib

import pytest
import torch
from torch.utils.data import TensorDataset

from jarvislm.model import GPT
from jarvislm.training.train import (
    REFERENCE_SAMPLE_PROMPTS,
    TARGET_TRAIN_TOKENS,
    TOKENS_PER_REFERENCE_STEP,
    TrainConfig,
    _prepare_data_if_requested,
    cosine_learning_rate,
    generate_qualitative_samples,
    smoke_test,
    train,
)


def test_reference_350m_profile_matches_the_single_h200_recipe() -> None:
    config = TrainConfig()

    assert config.model.vocab_size == 50_304
    assert config.model.max_seq_len == 1_024
    assert config.model.d_model == 1_024
    assert config.model.n_layers == 24
    assert config.model.n_heads == config.model.n_kv_heads == 16
    assert config.model.use_qk_norm is False
    assert config.model.use_diff_attn is False
    assert config.model.use_mhc is False
    assert config.recipe_name == "v1_350m"
    assert config.tokens_per_step == TOKENS_PER_REFERENCE_STEP == 524_288
    assert config.max_steps == 20_000
    assert TARGET_TRAIN_TOKENS == 9_933_989_297
    assert config.max_learning_rate == pytest.approx(3e-4)
    assert config.min_learning_rate == pytest.approx(3e-5)
    assert config.warmup_steps == 1_000
    assert config.weight_decay == pytest.approx(0.1)
    assert config.use_muon is False
    assert config.use_ema is False
    assert config.generate_samples is True
    assert config.sample_prompts == REFERENCE_SAMPLE_PROMPTS
    assert config.sample_max_new_tokens == 100
    assert config.sample_temperature == pytest.approx(0.8)
    assert config.required_gpu == "H200"


def test_reference_schedule_warms_up_and_decays() -> None:
    config = TrainConfig.smoke()
    config.max_steps = 10
    config.warmup_steps = 2
    config.min_learning_rate = 1e-4
    config.max_learning_rate = 1e-3

    assert cosine_learning_rate(config, 0) == pytest.approx(1e-4)
    assert cosine_learning_rate(config, 2) == pytest.approx(1e-3)
    assert cosine_learning_rate(config, 10) == pytest.approx(1e-4)


def test_cpu_smoke_test_runs_two_v1_adamw_updates() -> None:
    result = smoke_test()

    assert result.device == "cpu"
    assert len(result.metrics) == 2
    assert result.metrics[-1].step == 2
    assert result.metrics[-1].loss > 0
    assert result.metrics[-1].grad_norm > 0
    assert result.metrics[-1].step_time_ms > 0


def test_checkpoint_resume_continues_completed_step(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.checkpoint_dir = tmp_path / "checkpoints"
    config.max_steps = 1
    config.save_interval = 0
    config.use_muon = False
    config.validate()
    tokens = torch.randint(0, config.model.vocab_size, (4, config.model.max_seq_len + 1))
    dataset = TensorDataset(tokens[:, :-1], tokens[:, 1:])
    first = train(config, dataset)

    config.max_steps = 2
    config.validate()
    resumed = train(config, dataset)

    assert first.metrics[-1].step == 1
    assert resumed.metrics[-1].step == 2


def test_checkpoint_resume_skips_a_newer_unreadable_checkpoint(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.checkpoint_dir = tmp_path / "checkpoints"
    config.max_steps = 1
    config.save_interval = 1
    tokens = torch.randint(0, config.model.vocab_size, (4, config.model.max_seq_len + 1))
    dataset = TensorDataset(tokens[:, :-1], tokens[:, 1:])
    train(config, dataset)

    (config.checkpoint_dir / "last.pt").write_bytes(b"truncated checkpoint")
    config.max_steps = 2
    resumed = train(config, dataset)

    assert resumed.metrics[-1].step == 2


def test_fixed_prompt_generation_preserves_training_rng(monkeypatch) -> None:
    train_module = importlib.import_module("jarvislm.training.train")
    config = TrainConfig.smoke()
    model = GPT(config.model).train()

    def fake_generate_text(model, tokenizer, prompt, **kwargs):
        del model, tokenizer, kwargs
        return f"{prompt}:{torch.rand(1).item():.6f}"

    monkeypatch.setattr(train_module, "generate_text", fake_generate_text)
    state_before = torch.random.get_rng_state().clone()
    samples = generate_qualitative_samples(
        model,
        object(),  # type: ignore[arg-type]
        REFERENCE_SAMPLE_PROMPTS,
        max_new_tokens=100,
        temperature=0.8,
        top_k=None,
        seed=42,
    )

    assert tuple(prompt for prompt, _ in samples) == REFERENCE_SAMPLE_PROMPTS
    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert model.training is True


def test_data_preparation_requires_positive_token_budgets(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.data_dir = tmp_path / "train"
    config.val_dir = tmp_path / "val"

    with pytest.raises(ValueError, match="positive"):
        _prepare_data_if_requested(config, True, train_tokens=0, val_tokens=1)
