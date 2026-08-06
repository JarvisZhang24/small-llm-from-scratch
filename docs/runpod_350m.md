# RunPod 350M single-GPU workflow

JarvisLM follows the historical reference V1 recipe at source commit
`74351e3`: 353,502,208 parameters, GPT-2 `uint16` FineWeb-Edu `sample-10BT`
shards, full-parameter AdamW, no EMA, bf16, 20,000 updates, and 524,288 tokens
per update. The current complete sample contains 9,953,989,297 tokenized tokens;
JarvisLM reserves 20M for validation and uses 9,933,989,297 for training. Run
all cloud operations from a RunPod Pod with a network volume attached at
`/workspace`.

The source training narrative is documented in
[Building a 350M Transformer From Scratch](https://john463212.substack.com/p/building-a-350m-transformer-from).
JarvisLM follows its published V1 architecture and training recipe while
recording the validation-split difference described at the end of this guide.

Use one network-volume data center that can supply both the A100 preflight Pod
and the H200 full-training Pod. This project uses a 100 GB CA-MTL-3 network
volume. The volume persists after a Pod is terminated, but it must be selected
while creating every Pod.

## One-time volume setup

On the first inexpensive Pod, clone the repository and install dependencies:

```bash
git clone https://github.com/JarvisZhang24/small-llm-from-scratch.git \
  /workspace/jarvislm-350m/repo
cd /workspace/jarvislm-350m/repo
python -m pip install -e '.[dev]'
export HF_TOKEN='...'        # do not commit this value
export WANDB_API_KEY='...'   # omit only with JARVISLM_USE_WANDB=0
```

In the RunPod Pod configuration, attach the stored `HF_TOKEN` and
`WANDB_TOKEN` Secrets as environment variables. The scripts never print their
values. Before any work starts, they check that `HF_TOKEN` exists; A100/H200
stages also check W&B when it is enabled, and verify the expected GPU model.

The persistent layout is:

```text
/workspace/jarvislm-350m/
├── repo/
├── data/fineweb-edu-v1-sample-10bt/{train,val,manifest.json}
├── cache/{huggingface,wandb}
└── runs/
    ├── {v1-a100-preflight-mb16,v1-h200-10b}
    └── {v2-modern-a100-preflight-mb16,v2-modern-h200-5.5b}
```

Prepare the 9,933,989,297/20M token train/validation split exactly once:

```bash
scripts/runpod/run_350m.sh prepare
```

This creates `manifest.json` only after both split directories contain their
exact expected token counts. The `sample-10BT` name is approximate; requesting
10B training tokens in addition to validation exhausts the stream. If all
9,933,989,297 training tokens and 20M validation tokens were already flushed,
re-running `prepare` verifies them and creates the missing manifest without a
download. Other partial shard sets are rejected.

## A100 1,000-step preflight

Create an A100 Pod in the same data center, attach the same network volume,
then run:

```bash
cd /workspace/jarvislm-350m/repo
python -m pip install -e '.[dev]'
scripts/runpod/run_350m.sh preflight
```

This uses the source-faithful V1 batch configuration: `micro_batch=16`,
`grad_accumulation=32`, and 524,288 tokens per optimizer update. It runs 1,000
updates so the V1 linear warmup reaches approximately its peak `3e-4` learning
rate. It resumes from `runs/v1-a100-preflight-mb16/checkpoints/last.pt` after an
interruption and writes `train.log` plus `preflight_report.json` under
`runs/v1-a100-preflight-mb16/`. The separate directory prevents checkpoints
from the earlier conservative `2/256` diagnostic from entering this V1 gate.
At each 100-step preflight validation, it also emits the same fixed-prompt
qualitative samples used by the full run.

The report passes only when the 1,000-step checkpoint exists, at least 20 metric
records are present, losses and gradient norms are finite, the final logged
loss is below the first logged loss, and logged gradient norms remain in
`(0, 1000]`. Do not begin the H200 run if it fails.

## H200 full run

After a passed preflight, deploy an H200 Pod in the same network-volume data
center and run:

```bash
cd /workspace/jarvislm-350m/repo
python -m pip install -e '.[dev]'
scripts/runpod/run_350m.sh full
```

The H200 command refuses to start unless the A100 report explicitly passed. It
uses `micro_batch=32`, `grad_accumulation=16`, eight data-loader workers,
20,000 steps, and
`--required-gpu H200`. It writes checkpoints to
`runs/v1-h200-10b/checkpoints`; re-running it automatically resumes from the most
recent readable checkpoint. Checkpoints are written through a temporary file and
atomically renamed; if the newest file is truncated or unreadable, resume falls
back to the previous checkpoint instead of restarting from step zero. CPU and
CUDA RNG states are restored on their required devices.

Every 500 steps, the full run reports validation loss and perplexity, then
generates 100-token completions at temperature `0.8` from the four fixed V1
prompts used in the reference write-up: `The meaning of life is`,
`In a distant galaxy,`, `def fibonacci(n):`, and
`The president announced that`. The sampling RNG is isolated from training and
reset to a fixed seed, so changes between checkpoints reflect model learning
rather than different random draws. Metrics and samples are sent to W&B and the
complete console stream is appended to `runs/v1-h200-10b/train.log`.

Use `JARVISLM_USE_WANDB=0` only when W&B is intentionally disabled. To change
the volume root, set `JARVISLM_VOLUME_ROOT`; to change only a planned test
length, set `JARVISLM_PREFLIGHT_STEPS` or `JARVISLM_H200_STEPS`. Changing either
value makes the run a diagnostic override rather than the recorded V1 recipe.

The reference repository did not publish how its validation directory was
constructed in code. The later reference write-up describes two held-out shards
(approximately 2%) but does not identify which shards were selected. JarvisLM
therefore keeps its already recorded, deterministic extension: it reserves a
disjoint 20M-token validation split and trains on the remaining 9.934B tokens.
This is sufficient for the fixed 20-batch validation sample but is not the same
split as the write-up, so do not compare absolute validation loss directly with
the source model's reported value.

## V2 modern matched-budget comparison

The `v2_modern` recipe keeps every controlled variable from the V1 H200 run
that can be held fixed: the exact train/validation shards, GPT-2 tokenizer,
seed 42, context 1,024, global batch of 524,288 tokens, validation batches,
fixed prompts, and the original 20,000-step learning-rate trajectory. It
changes only the intended V2 stack:

- 16 query heads / 4 KV heads (GQA);
- QK-Norm and Differential Attention;
- Muon for eligible attention/MLP matrices plus AdamW for embeddings, norms,
  and other parameters;
- EMA with decay 0.9995, used for the primary validation and samples;
- no mHC and no XSA.

The V2 run stops at the same 10,500 updates as the recorded V1 baseline. This
is 5,505,024,000 scheduled tokens. Its cosine schedule still uses 20,000 as
the decay horizon, so the learning rate at every compared step matches V1;
10,500 is an early-stop boundary, not a new cosine endpoint.

First run the short, recipe-specific A100 gate:

```bash
cd /workspace/jarvislm-350m/repo
git pull
python -m pip install -e '.[dev]'
scripts/runpod/run_v2_modern.sh preflight
```

This runs 200 updates at `micro_batch=16`, `grad_accumulation=32` and writes
only under `runs/v2-modern-a100-preflight-mb16`. It does not read or overwrite
any V1 checkpoint. After its report passes, launch the H200 comparison:

```bash
scripts/runpod/run_v2_modern.sh full
```

The full command uses `micro_batch=32`, `grad_accumulation=16`, stops at
10,500, and resumes only from `runs/v2-modern-h200-5.5b/checkpoints`. Every
500 steps it records both `validation/raw_*` and `validation/ema_*`; the
backward-compatible `validation/loss` and `validation/perplexity` curves point
to EMA for this recipe. Fixed-prompt generations also use EMA weights.

Before leaving a new H200 run unattended, inspect its first 50-100 steps for
finite loss/gradients, stable throughput, and memory headroom. The A100 gate
checks correctness and recovery; it cannot predict the exact H200 throughput.
