import torch

from jarvislm.model import GPT, ModelConfig
from jarvislm.optim.muon import Muon, configure_optimizers, newton_schulz


def test_newton_schulz_keeps_a_zero_gradient_finite() -> None:
    gradient = torch.zeros(4, 8)

    update = newton_schulz(gradient)

    assert torch.equal(update, gradient)
    assert torch.isfinite(update).all()


def test_muon_step_updates_a_tall_matrix_without_changing_shape() -> None:
    parameter = torch.nn.Parameter(torch.randn(8, 4))
    parameter.grad = torch.randn_like(parameter)
    before = parameter.detach().clone()

    Muon([parameter], lr=1e-2).step()

    assert parameter.shape == before.shape
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, before)


def test_muon_and_adamw_partition_every_trainable_parameter_once() -> None:
    model = GPT(
        ModelConfig(
            vocab_size=128,
            max_seq_len=16,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden_dim=128,
            use_flash=False,
            use_qk_norm=True,
            use_diff_attn=True,
            use_mhc=False,
        )
    )

    muon, adamw = configure_optimizers(model, lr=3e-4, muon_lr=1.5e-4, weight_decay=0.1)
    muon_parameters = {
        id(parameter) for group in muon.param_groups for parameter in group["params"]
    }
    adamw_parameters = {
        id(parameter) for group in adamw.param_groups for parameter in group["params"]
    }
    trainable_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    assert muon_parameters
    assert adamw_parameters
    assert muon_parameters.isdisjoint(adamw_parameters)
    assert muon_parameters | adamw_parameters == trainable_parameters
    assert id(model.token_embeddings.weight) in adamw_parameters
    assert id(model.blocks[0].attention.W_q.weight) in muon_parameters
