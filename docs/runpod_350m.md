# RunPod 350M single-GPU workflow

JarvisLM follows the historical reference V1 recipe at source commit
`74351e3`: 353,502,208 parameters, GPT-2 `uint16` FineWeb-Edu `sample-10BT`
shards, full-parameter AdamW, no EMA, bf16, 20,000 updates, and 524,288 tokens
per update. The current complete sample contains 9,953,989,297 tokenized tokens;
JarvisLM reserves 20M for validation and uses 9,933,989,297 for training. Run
all cloud operations from a RunPod Pod with a network volume attached at
`/workspace`.

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
└── runs/{v1-a100-preflight,v1-h200-10b}
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

This preserves the full 350M architecture, AdamW optimizer, and 524,288-token
effective batch with `micro_batch=2` and `grad_accumulation=256`. It runs 1,000
updates so the V1 linear warmup reaches approximately its peak `3e-4` learning
rate. It resumes from `runs/v1-a100-preflight/checkpoints/last.pt` after an
interruption and writes `train.log` plus `preflight_report.json` under
`runs/v1-a100-preflight/`.

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
uses `micro_batch=16`, `grad_accumulation=32`, 20,000 steps, and
`--required-gpu H200`. It writes checkpoints to
`runs/v1-h200-10b/checkpoints`; re-running it automatically resumes from the most
recent checkpoint.

Use `JARVISLM_USE_WANDB=0` only when W&B is intentionally disabled. To change
the volume root, set `JARVISLM_VOLUME_ROOT`; to change only a planned test
length, set `JARVISLM_PREFLIGHT_STEPS` or `JARVISLM_H200_STEPS`. Changing either
value makes the run a diagnostic override rather than the recorded V1 recipe.

The reference repository did not publish how its validation directory was
constructed. JarvisLM therefore records a transparent extension: it reserves a
disjoint 20M-token validation split and trains on the remaining 9.934B tokens. Do
not compare its absolute validation loss directly with an undocumented source
split.
