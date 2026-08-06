# JarvisLM

> A decoder-only language model built from core components, with a reproducible pre-training and inference-optimization roadmap.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)
![Status](https://img.shields.io/badge/status-active%20development-f59e0b)

## Overview

JarvisLM is a learning-driven systems project that rebuilds a small decoder-only Transformer step by step, then scales the implementation to a roughly 350M-parameter pre-training run. The goal is not to wrap an existing LLM trainer: it is to understand, implement, test, and benchmark the core pieces of an LLM training stack.

The project starts with individually testable model components, then adds data sharding, distributed training, checkpoint recovery, evaluation, and inference optimization.

## Why this project

Training a language model is an end-to-end engineering problem. A reliable implementation needs more than a Transformer forward pass:

- correct tensor shapes and causal masking;
- numerically stable normalization and mixed-precision behavior;
- streaming tokenized data and reproducible train/validation splits;
- optimizer, learning-rate schedule, checkpointing, and restart recovery;
- experiment tracking, evaluation, and latency/throughput measurement.

JarvisLM treats each of these as a separately verifiable component.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Python package and editable installation | Complete | `src/` package layout with `pyproject.toml` |
| Model and attention components | Complete | RMSNorm, SwiGLU, RoPE, causal attention, MHA/GQA options |
| Full GPT model and loss | Complete | Unit-tested forward, generation, gradients, and parameter count |
| Tokenization and binary shards | Complete | tiktoken GPT-2 and `uint16` FineWeb-Edu shards |
| Training and checkpoint recovery | Complete | V1 AdamW baseline, bf16, cosine schedule, W&B, resume; Muon/EMA optional |
| RunPod training workflow | Complete | CA-MTL-3 network volume, A100 gate, isolated H200 V1 run |
| 350M pre-training results | Pending | Must be measured on the user's actual RunPod execution |
| Muon / AdamW ablation | Pending | Requires fixed-data controlled experiment artifacts |
| KV cache and inference benchmark | In progress | Compare cached and uncached decoding |

Only completed items are presented as completed. Training results and benchmark numbers will be added after reproducible runs are available.

## Target architecture

The target experimental configuration follows a modern decoder-only Transformer design:

| Hyperparameter | Target value | Purpose |
| --- | ---: | --- |
| Vocabulary size | 50,304 | Padded GPT-2 tokenizer vocabulary |
| Context length | 1,024 | Maximum training and generation context |
| Hidden size (`d_model`) | 1,024 | Token representation width |
| Transformer layers | 24 | Model depth |
| Query heads | 16 | Multi-head attention |
| KV heads | 16 | Reference V1 baseline uses standard MHA (GQA is optional) |
| Head dimension | 64 | `d_model / n_heads` |
| FFN hidden dimension | 2,730 | `int(8 / 3 * d_model)` for SwiGLU |
| Normalization | RMSNorm | Pre-normalization Transformer blocks |
| Position encoding | RoPE | Rotary positional embeddings |
| Attention backend | PyTorch SDPA | Causal attention with optimized kernels where available |

The final parameter count will be calculated and recorded from the implemented model rather than claimed in advance. The approximately 350M target depends on the final attention and weight-tying choices.

```text
Token IDs
   │
   ▼
Token Embedding
   │
   ▼
Transformer Block × 24
   ├── RMSNorm → MHA + RoPE + causal attention → residual
   └── RMSNorm → SwiGLU → residual
   │
   ▼
Final RMSNorm → tied LM head → vocabulary logits
```

## Repository structure

```text
.
├── src/
│   └── jarvislm/
│       ├── __init__.py
│       └── model/
│           ├── config.py       # Model architecture configuration
│           ├── rmsnorm.py      # RMSNorm implementation
│           ├── swiglu.py       # SwiGLU implementation (in progress)
│           ├── rope.py         # Rotary position embeddings (planned)
│           ├── attention.py    # Causal GQA attention (planned)
│           ├── block.py        # Transformer block (planned)
│           └── gpt.py          # Decoder-only language model (planned)
├── tests/                      # Component-level unit tests
├── configs/                    # Training YAML configurations (planned)
├── scripts/                    # Data, training, evaluation commands (planned)
├── docs/                       # Experiment reports and figures (planned)
└── pyproject.toml
```

## Quick start

### 1. Create the environment

```bash
conda create -n jarvislm python=3.11 pip -y
conda activate jarvislm
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Run tests

```bash
pytest -v
ruff check .
```

### 3. Check local acceleration on Apple Silicon

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

`True` means the local Mac can be used for unit tests and small-scale smoke runs. Long pre-training runs will use a CUDA cloud GPU.

## Implemented components

### ModelConfig

`ModelConfig` is the single source of truth for architecture parameters. It validates relationships that would otherwise fail later inside tensor operations:

- `d_model` must be divisible by `n_heads`;
- `n_heads` must be divisible by `n_kv_heads` for GQA;
- the per-head dimension must be even for RoPE;
- dropout, epsilon, vocabulary size, and context length must be valid.

### RMSNorm

For an input vector \(x\) along the hidden dimension:

\[
\operatorname{RMS}(x) = \sqrt{\operatorname{mean}(x^2) + \epsilon}
\]

\[
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\operatorname{RMS}(x)}
\]

The implementation is covered by tests for output shape, the manual formula, gradients, and invalid arguments.

## Development and verification plan

The implementation order is intentional. Each stage has an observable correctness check before the next stage is started.

1. `ModelConfig` and package installation — configuration tests pass.
2. RMSNorm — manual-formula and gradient tests pass.
3. SwiGLU — manual-formula and shape tests pass.
4. RoPE — rotation-preserves-norm test passes.
5. Causal attention — output-shape test and future-token leakage test pass.
6. Transformer block and GPT — one-batch overfitting loss drops below a predefined threshold.
7. Data pipeline — token shard boundaries and deterministic validation split are verified.
8. Trainer — checkpoint save/resume reproduces the next training step.
9. Small-model smoke run — local MPS/CUDA loss curve and generated samples are recorded.
10. Cloud pre-training — fixed configuration, token budget, checkpoints, and W&B run are published.
11. Ablations — compare optimizer and attention/inference choices under controlled settings.

## Planned experiments

The following experiments will be reported with the exact configuration, hardware, seed, token budget, and command used.

| Experiment | Comparison | Metrics |
| --- | --- | --- |
| Optimizer | AdamW vs. Muon | validation loss, tokens/s, stability |
| Attention | MHA vs. GQA | parameter count, memory, tokens/s, validation loss |
| Inference | cached vs. uncached decoding | prefill latency, decode tokens/s, peak memory |
| Context | 512 vs. 1,024 tokens | throughput and validation loss |

## Reproducibility principles

- Keep model, data, and training parameters in version-controlled configuration files.
- Record random seed, tokenizer version, dataset revision, total tokens, hardware, and software versions.
- Store model state, optimizer state, scheduler state, step, and random-number-generator state in checkpoints.
- Publish training curves and generated examples alongside the corresponding checkpoint.
- Never report benchmark, speed, or parameter-count claims without a script and a run artifact.

## Compute plan

- **Mac (MPS):** unit tests and small, local component checks only.
- **RunPod network volume:** persistent FineWeb-Edu shards, checkpoints, Hugging Face cache, and W&B logs.
- **RunPod A100:** 1,000-step full-architecture V1 preflight through warmup.
- **RunPod H200:** fresh 20,000-step V1 run after the A100 report passes; interrupted H200 runs resume in place.

The exact operational commands and failure safeguards are in
[the RunPod 350M guide](docs/runpod_350m.md).

## Resume-ready project description

Use this only after the claimed parts have been completed and documented:

```text
JarvisLM: Built a decoder-only language-model training stack from core PyTorch
components; implemented RMSNorm, RoPE, SwiGLU, causal/GQA attention, token
sharding, checkpoint recovery, and inference KV caching. Trained and evaluated
an approximately 350M-parameter model, and benchmarked optimizer and decoding
trade-offs with reproducible experiment artifacts.
```

## References and acknowledgements

- Zhang & Sennrich, *Root Mean Square Layer Normalization* (RMSNorm).
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU).
- Su et al., *RoFormer* (RoPE).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW).

This repository is an independent educational implementation. When external repositories, papers, datasets, or code patterns are used as references, they will be credited in the relevant source file and experiment report.

## License

License selection is pending. Do not reuse the code as a dependency until a license file is added.
