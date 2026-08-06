# JarvisLM experiment results

## V1 baseline: H200 early stop at 10,500 updates

This is a measured user run, not a result copied from the reference project.

| Field | Value |
| --- | ---: |
| Recipe | `v1_350m` |
| Parameters | 353,502,208 |
| Hardware | 1x NVIDIA H200 SXM |
| Context | 1,024 tokens |
| Tokens/update | 524,288 |
| Completed updates | 10,500 |
| Scheduled tokens processed | 5,505,024,000 |
| Optimizer | AdamW |
| EMA | Off |
| Validation loss at step 10,500 | 2.9440 |
| Validation perplexity at step 10,500 | 18.99 |
| Typical throughput | approximately 164k tokens/s |

The run was intentionally stopped at 10,500 to reserve compute budget for the
V2 and SFT stages. It is therefore a 5.5B-token checkpoint, not a completed
10B-token V1 training claim.

## V2 modern comparison protocol

V2 will be compared at exactly 10,500 updates with the same data split, seed,
context, tokens/update, validation protocol, and 20,000-step LR schedule used
by V1. The treatment combines GQA, QK-Norm, Differential Attention, Muon, and
EMA. Raw and EMA validation metrics are recorded separately so EMA's marginal
effect remains visible.

V2 results remain pending until the RunPod artifacts are produced.
