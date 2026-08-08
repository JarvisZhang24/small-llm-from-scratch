---
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- causal-lm
- pretraining
- from-scratch
- gqa
- differential-attention
- muon
datasets:
- HuggingFaceFW/fineweb-edu
---

# JarvisLM-350M

A decoder-only language model implemented from core PyTorch modules — no
off-the-shelf trainer — and pre-trained from scratch on a single GPU.

These are the **V2 Modern EMA** weights: 315,758,848 parameters, 10,500
optimizer updates, 5,505,024,000 scheduled tokens of FineWeb-Edu.

- Code, training recipe, and full experiment write-up:
  [JarvisZhang24/small-llm-from-scratch](https://github.com/JarvisZhang24/small-llm-from-scratch)
- Training telemetry:
  [public W&B report](https://api.wandb.ai/links/jarviszhang-new-york-university/ngem6azk)

## What this is

The repository runs a controlled comparison between two models trained on the
same data, seed, token budget, and learning-rate trajectory:

| | V1 Base | V2 Modern (this model) |
| --- | ---: | ---: |
| Parameters | 353,502,208 | 315,758,848 |
| Attention | MHA, 16Q/16KV | GQA 16Q/4KV + Differential Attention |
| QK-Norm | Off | On |
| Optimizer | AdamW | Muon + AdamW |
| EMA | Off | On, decay 0.9995 |

Both were trained for 10,500 updates at a 524,288-token global batch on one
NVIDIA H200 SXM, with 24 layers, hidden size 1,024, and a 1,024-token context.

## Results

| Final checkpoint | V1 Base | V2 Raw | V2 EMA (this model) |
| --- | ---: | ---: | ---: |
| Updates | 10,586 | 10,500 | 10,500 |
| Validation loss | 2.9595 | 2.9139 | **2.8934** |
| Validation perplexity | 19.29 | 18.43 | **18.05** |

V2 reached a lower validation loss than V1 with 10.7% fewer parameters: 4.5%
lower perplexity for the raw weights, 6.4% for these EMA weights.

All three figures come from one pass over the complete 20M-token held-out
split — the same 19,520 sequences at one fixed batch size — using
`scripts/evaluate_checkpoints.py` from the repository. V1's saved checkpoint
carries 86 updates more than V2's, so the measured gap understates rather than
flatters the V2 architecture.

## Architecture

- Decoder-only, pre-normalization Transformer; token embedding tied to the LM head
- RMSNorm, RoPE, SwiGLU feed-forward
- Grouped-query attention (16 query heads, 4 KV heads) with QK-Norm
- Differential Attention: two causal attention passes combined with a learned,
  layer-dependent coefficient
- GPT-2 tokenizer (50,257 usable IDs); the model vocabulary is padded to 50,304
  logits for hardware-friendly dimensions

## Usage

The architecture is custom, so it does not load through `transformers`. Clone
the repository and load the weights directly:

```python
import json
from pathlib import Path

from safetensors.torch import load_file

from jarvislm.model import GPT, ModelConfig

fields = ModelConfig.__dataclass_fields__
metadata = json.loads(Path("config.json").read_text())
config = ModelConfig(**{k: v for k, v in metadata.items() if k in fields})

state = load_file("model.safetensors")
if config.tie_weights:
    # The LM head shares the embedding table, so it is not stored separately.
    state["lm_head.weight"] = state["token_embeddings.weight"]

model = GPT(config).eval()
model.load_state_dict(state)
```

Sampling must go through `jarvislm.inference.generate_token_ids`, which excludes
the padded vocabulary entries that the tokenizer cannot decode:

```python
from jarvislm.inference import generate_text
from jarvislm.tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer()
print(generate_text(model, tokenizer, "The meaning of life is", max_new_tokens=100))
```

## Limitations

This is a **base model** trained on 5.505B tokens. It has had no instruction
tuning, no RLHF, and no safety alignment.

- It completes text; it does not follow instructions.
- Factual claims in its output are frequently wrong. It will confidently invent
  names, dates, and events.
- Code generation is not reliable.
- Training data is FineWeb-Edu, an English web corpus. Expect web-scale social
  biases in the output, and poor performance outside English.
- No benchmark evaluation (`lm-eval-harness` or similar) has been run, so no
  claims are made about downstream task performance.

Do not deploy this model in any setting where incorrect output carries a cost.

## Training data

FineWeb-Edu `sample-10BT`, tokenized with the GPT-2 tokenizer and written to
`uint16` shards. A deterministic 20,000,000-token split is held out for
validation; training draws from the remaining pool. This model consumed
5,505,024,000 scheduled tokens, which is a fraction of the prepared corpus.

## Attribution

An independent educational reproduction, following the write-ups and the
[`modern-llm`](https://github.com/JohnEnev/modern-llm) repository by John Enev
as a behavioral reference. Only locally measured checkpoints are reported here;
no reference-project metric is presented as a JarvisLM result.
