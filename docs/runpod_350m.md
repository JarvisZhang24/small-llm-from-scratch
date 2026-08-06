# RunPod 350M single-GPU workflow

JarvisLM follows the reference V1 recipe: 353,502,208 parameters, GPT-2
`uint16` FineWeb-Edu shards, Muon plus AdamW, EMA, bf16, and 524,288 tokens
per optimizer update. Run all cloud operations from a RunPod Pod with a
network volume attached at `/workspace`.

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
├── data/fineweb-edu/{train,val,manifest.json}
├── cache/{huggingface,wandb}
└── runs/{a100-preflight,h200-10b}
```

Prepare the 10B/20M token train/validation split exactly once:

```bash
scripts/runpod/run_350m.sh prepare
```

This creates `manifest.json` only after both split directories contain their
exact expected token counts. If a Pod ends during preparation, the script
refuses to mix partial shards with a new run; inspect or back up that data
before manually removing it and restarting preparation.

## A100 500-step preflight

Create an A100 Pod in the same data center, attach the same network volume,
then run:

```bash
cd /workspace/jarvislm-350m/repo
python -m pip install -e '.[dev]'
scripts/runpod/run_350m.sh preflight
```

This preserves the full 350M architecture and 524,288-token effective batch
with `micro_batch=2` and `grad_accumulation=256`. It runs 500 updates, resumes
from `runs/a100-preflight/checkpoints/last.pt` after an interruption, and
writes `train.log` plus `preflight_report.json` under `runs/a100-preflight/`.

The report passes only when the 500-step checkpoint exists, at least 20 metric
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
uses `micro_batch=16`, `grad_accumulation=32`, 19,074 steps, and
`--required-gpu H200`. It writes checkpoints to
`runs/h200-10b/checkpoints`; re-running it automatically resumes from the most
recent checkpoint.

Use `JARVISLM_USE_WANDB=0` only when W&B is intentionally disabled. To change
the volume root, set `JARVISLM_VOLUME_ROOT`; to change only a planned test
length, set `JARVISLM_PREFLIGHT_STEPS` or `JARVISLM_H200_STEPS`.
