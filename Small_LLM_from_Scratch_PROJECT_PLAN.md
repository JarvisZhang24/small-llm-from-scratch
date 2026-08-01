# Small LLM from Scratch 项目实施方案

> 项目仓库：`small-llm-from-scratch`  
> 正式模型名称：`JarvisLM-120M`  
> 项目定位：从零实现、预训练、评测并优化一个小型 Decoder-only 语言模型。

---

## 1. 项目目标

本项目的目标不是复制某个仓库的全部代码，而是参考现代 LLM 的通用设计，独立完成一套可以训练、评测、恢复和优化的小型语言模型系统。

最终需要完成：

- 使用 PyTorch 从零实现 Decoder-only Transformer。
- 实现 RoPE、RMSNorm、SwiGLU 和 Causal Attention。
- 构建文本流式处理、tokenization 和二进制数据分片流程。
- 支持混合精度、梯度累积、学习率调度和断点续训。
- 在 Mac 上完成小模型验证。
- 在 Modal/RunPod 上训练约 120M 参数的正式模型。
- 完成 AdamW 与 Muon 对照实验。
- 实现并评测 KV Cache 推理加速。
- 发布代码、训练曲线、实验结果和模型权重。

---

## 2. 非目标

第一版暂时不做：

- 不直接训练 350M 或 672M 模型。
- 不追求和商业模型比较生成质量。
- 不同时实现所有实验架构。
- 不在第一阶段实现 Differential Attention、XSA、mHC。
- 不首先实现自己的 BPE tokenizer。
- 不首先进行大规模 SFT、PPO、DPO 或 GRPO。

这些内容可以作为后续扩展，但不能影响主线项目完成。

---

## 3. 核心研究问题

### 3.1 从零训练是否正确

- 模型能否过拟合单个 batch？
- 训练 loss 和 validation loss 是否正常下降？
- checkpoint 恢复后训练曲线能否连续？
- 生成过程是否严格满足因果性？

### 3.2 Muon 是否优于 AdamW

在固定以下条件时进行比较：

- 模型结构
- 初始化种子
- 数据顺序
- 训练 token 数量
- effective batch size
- 学习率搜索范围

比较指标：

- validation loss
- perplexity
- 收敛速度
- tokens/s
- 峰值显存
- 训练稳定性

### 3.3 KV Cache 能带来多大推理提升

使用同一套模型权重比较：

- Naive autoregressive decoding
- KV Cache decoding

比较指标：

- 首 token 延迟
- 每秒生成 tokens
- 不同上下文长度下的延迟
- KV Cache 显存占用
- 缓存与非缓存输出一致性

---

## 4. 模型规模规划

项目使用三个模型配置。

### 4.1 单元测试模型

用于快速测试模块正确性。

```yaml
name: jarvislm-test
vocab_size: 1024
d_model: 128
n_layers: 2
n_heads: 4
n_kv_heads: 4
max_seq_len: 64
dropout: 0.0
tie_embeddings: true
```

目标：

- CPU 上几秒内完成测试。
- 验证 shape、梯度、mask 和 checkpoint。

### 4.2 Mac 冒烟模型

用于验证完整训练流程。

```yaml
name: jarvislm-smoke
vocab_size: 50304
d_model: 256
n_layers: 6
n_heads: 8
n_kv_heads: 8
max_seq_len: 256
dropout: 0.0
tie_embeddings: true
```

预计约 18M 参数。

目标：

- 在 Mac CPU 或 MPS 上运行。
- 完成单 batch 过拟合。
- 完成小语料训练。
- 验证生成和 checkpoint 恢复。

### 4.3 云端正式模型

```yaml
name: jarvislm-120m
vocab_size: 50304
d_model: 768
n_layers: 12
n_heads: 12
n_kv_heads: 12
max_seq_len: 1024
mlp_hidden_size: 2048
dropout: 0.0
tie_embeddings: true
```

预计约 124M 参数。

正式名称：

```text
JarvisLM-120M
```

第一轮目标训练量：

```text
1B tokens
```

如果成本和结果合理，再增加到：

```text
2B tokens
```

---

## 5. 技术栈

```text
Python
PyTorch
tiktoken
NumPy
Hugging Face Datasets
Weights & Biases
pytest
ruff
Modal
RunPod
GitHub Actions
```

设备支持顺序：

```text
CUDA > MPS > CPU
```

参考资料：

- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Modal 长时间可恢复训练](https://modal.com/docs/examples/long-training)
- [RunPod 持久化存储](https://docs.runpod.io/pods/storage/types)

---

## 6. 项目目录结构

```text
small-llm-from-scratch/
├── .github/
│   └── workflows/
│       └── tests.yml
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   └── samples/
├── configs/
│   ├── test.yaml
│   ├── smoke.yaml
│   ├── jarvislm_120m.yaml
│   ├── adamw_ablation.yaml
│   └── muon_ablation.yaml
├── docs/
│   ├── architecture.md
│   ├── data.md
│   ├── experiment_log.md
│   ├── results.md
│   └── cloud_training.md
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── benchmark_inference.py
│   └── upload_model.py
├── src/
│   └── jarvislm/
│       ├── __init__.py
│       ├── config.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── rmsnorm.py
│       │   ├── swiglu.py
│       │   ├── rope.py
│       │   ├── attention.py
│       │   ├── block.py
│       │   ├── gpt.py
│       │   └── kv_cache.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── tokenizer.py
│       │   ├── shard_writer.py
│       │   ├── dataset.py
│       │   └── manifest.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── checkpoint.py
│       │   ├── schedule.py
│       │   ├── optimizers.py
│       │   └── distributed.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── validation.py
│       │   └── harness.py
│       └── inference/
│           ├── __init__.py
│           ├── sampling.py
│           └── benchmark.py
├── tests/
│   ├── fixtures/
│   ├── test_rmsnorm.py
│   ├── test_swiglu.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_block.py
│   ├── test_gpt.py
│   ├── test_dataset.py
│   ├── test_checkpoint.py
│   └── test_kv_cache.py
├── .gitignore
├── LICENSE
├── PROJECT_PLAN.md
├── README.md
└── pyproject.toml
```

不提交以下文件：

```text
data/
checkpoints/
wandb/
*.bin
*.pt
*.pth
*.safetensors
__pycache__/
.venv/
```

---

# 7. 阶段一：初始化项目

## 7.1 初始化 Git

```bash
git init -b main
```

## 7.2 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

## 7.3 安装基础依赖

```bash
pip install \
  torch \
  numpy \
  tiktoken \
  datasets \
  tqdm \
  pyyaml \
  wandb \
  pytest \
  ruff
```

## 7.4 检查设备

```bash
python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('MPS:', torch.backends.mps.is_available()); \
print('CUDA:', torch.cuda.is_available())"
```

## 7.5 验收标准

- [ ] 虚拟环境可以激活。
- [ ] PyTorch 可以导入。
- [ ] `pytest` 可以运行。
- [ ] Git 仓库初始化完成。
- [ ] `.gitignore` 已配置。
- [ ] 完成第一次 commit。

建议 commit：

```text
chore: initialize project structure
```

---

# 8. 阶段二：配置和设备抽象

创建：

```text
src/jarvislm/config.py
```

实现：

- `ModelConfig`
- `TrainConfig`
- `DataConfig`
- YAML 加载
- 配置验证
- 配置保存

配置必须包含：

```python
vocab_size
d_model
n_layers
n_heads
n_kv_heads
max_seq_len
mlp_hidden_size
dropout
tie_embeddings
use_flash_attention
```

设备选择逻辑：

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

验收标准：

- [ ] YAML 能正确加载。
- [ ] 非法 head 数量会报错。
- [ ] `d_model % n_heads == 0`。
- [ ] `n_heads % n_kv_heads == 0`。
- [ ] 配置可以序列化到 checkpoint。
- [ ] CPU/MPS/CUDA 选择逻辑经过测试。

建议 commit：

```text
feat: add model and training configuration
```

---

# 9. 阶段三：实现 Transformer 基础模块

## 9.1 RMSNorm

文件：

```text
src/jarvislm/model/rmsnorm.py
tests/test_rmsnorm.py
```

需要验证：

- [ ] 输入输出 shape 一致。
- [ ] 输出为有限数值。
- [ ] 输入梯度存在。
- [ ] scale 参数梯度存在。
- [ ] 支持 float32 和混合精度。

建议 commit：

```text
feat: implement rmsnorm with tests
```

## 9.2 SwiGLU

实现：

```text
SwiGLU(x) = down(SiLU(gate(x)) * up(x))
```

需要验证：

- [ ] 输出维度正确。
- [ ] bias 配置正确。
- [ ] backward 正常。
- [ ] 参数量符合计算结果。

建议 commit：

```text
feat: implement swiglu feed-forward network
```

## 9.3 RoPE

实现内容：

- 频率表生成
- cos/sin cache
- Q/K 旋转
- position offset
- device/dtype 转换

需要验证：

- [ ] RoPE 前后向量范数基本不变。
- [ ] 不同 position 的旋转结果不同。
- [ ] 相同 position 的结果可复现。
- [ ] 支持缓存推理时的 position offset。
- [ ] CPU/MPS/CUDA 行为一致。

建议 commit：

```text
feat: implement rotary position embeddings
```

## 9.4 Causal Attention

第一版实现普通 MHA：

```text
Q = XWq
K = XWk
V = XWv
A = softmax(QK^T / sqrt(d_head) + causal_mask)
Y = AVO
```

暂时不要先做 GQA。

需要验证：

- [ ] 输出 shape 正确。
- [ ] causal mask 正确。
- [ ] 不会读取未来 token。
- [ ] backward 正常。
- [ ] 输出没有 NaN。

未来泄漏测试：

1. 构造两个完全相同的输入。
2. 只修改最后一个 token。
3. 比较最后位置之前的输出。
4. 之前的输出必须保持一致。

普通实现正确后，再加入：

```python
torch.nn.functional.scaled_dot_product_attention
```

然后比较普通 Attention 和 SDPA 输出。

建议 commit：

```text
feat: implement causal multi-head attention
```

## 9.5 Transformer Block

结构：

```python
x = x + attention(rmsnorm_1(x))
x = x + mlp(rmsnorm_2(x))
```

需要验证：

- [ ] shape 保持不变。
- [ ] residual 正确。
- [ ] 梯度能到达输入和各层参数。
- [ ] 多层堆叠不会出现 NaN。

建议 commit：

```text
feat: implement pre-norm transformer block
```

## 9.6 完整 GPT

包含：

- Token Embedding
- Transformer Blocks
- Final RMSNorm
- LM Head
- Embedding/LM Head 权重共享
- Cross-entropy loss

需要验证：

- [ ] logits shape 为 `[B, T, vocab_size]`。
- [ ] targets 存在时返回 loss。
- [ ] 权重共享使用同一个参数对象。
- [ ] backward 正常。
- [ ] 参数量符合配置。
- [ ] padded vocab IDs 不会在生成时被采样。

建议 commit：

```text
feat: implement decoder-only language model
```

---

# 10. 阶段四：单 Batch 过拟合测试

这是第一个强制质量门槛。

## 10.1 测试方法

准备一小段固定文本，将同一个 batch 重复训练数百次。

训练设置：

```yaml
batch_size: 4
seq_len: 64
learning_rate: 0.001
max_steps: 500
weight_decay: 0.0
```

## 10.2 验收标准

- [ ] 初始 loss 接近随机预测水平。
- [ ] loss 持续下降。
- [ ] 最终 loss 可以降到 1 以下。
- [ ] 模型可以复述或补全文本。
- [ ] 无 NaN。
- [ ] 梯度范数正常。

如果不能过拟合，停止后续工作，检查：

- target shift
- causal mask
- RoPE
- loss flatten
- weight tying
- optimizer
- learning rate

建议 commit：

```text
test: add single-batch overfitting validation
```

---

# 11. 阶段五：实现数据管线

## 11.1 Tokenizer

正式训练第一版使用 GPT-2 `tiktoken`。

```python
enc = tiktoken.get_encoding("gpt2")
```

模型词表：

```text
真实词表：50257
填充后词表：50304
```

生成时必须屏蔽：

```text
token_id >= 50257
```

避免采样到无法解码的填充 token。

## 11.2 文档编码

每篇文档：

```text
encode(text) + [EOT]
```

输出：

```text
uint16 token IDs
```

## 11.3 Shard Writer

实现：

- 固定 shard token 数量
- 流式写入
- 不一次性将全部数据放入内存
- 最后一个不足完整大小的 shard 也能保存
- 保存总 token 数量和文档数量

建议初始 shard：

```text
10M tokens/shard
```

正式训练可使用：

```text
50M-100M tokens/shard
```

## 11.4 Dataset

每个样本读取：

```python
tokens = shard[start:start + seq_len + 1]
input_ids = tokens[:-1]
targets = tokens[1:]
```

需要验证：

- [ ] 输入和目标均为 `torch.long`。
- [ ] shape 正确。
- [ ] shift 正确。
- [ ] 不会返回不完整序列。
- [ ] 不会错误跨越 shard。
- [ ] train/validation shard 不重叠。

## 11.5 Manifest

每个 shard 记录：

```json
{
  "path": "data/train/shard_0001.bin",
  "split": "train",
  "source": "fineweb_edu",
  "tokens": 10000000,
  "sha256": "..."
}
```

manifest 必须固定：

- 数据来源
- train/validation split
- shard token 数量
- 校验哈希

建议 commit：

```text
feat: add streaming tokenization and shard dataset
```

---

# 12. 阶段六：实现训练系统

## 12.1 基础训练循环

第一版仅实现：

- AdamW
- forward
- backward
- optimizer step
- loss logging

确认正确后再逐步加入其他功能。

## 12.2 学习率

实现：

```text
linear warmup + cosine decay
```

记录：

```text
train/loss
train/lr
train/grad_norm
train/tokens_per_second
train/step_time
```

## 12.3 梯度累积

保证：

```text
effective_batch_tokens
= micro_batch_size
× grad_accum_steps
× seq_len
× world_size
```

loss 必须除以：

```text
grad_accum_steps
```

## 12.4 混合精度

CUDA 上使用：

```text
bf16 autocast
```

Mac MPS 如果算子不稳定，可以先使用 float32 完成功能验证。

## 12.5 Validation

每隔固定步数：

- 切换 `model.eval()`
- 禁用梯度
- 使用固定 validation batches
- 计算平均 loss
- 计算 perplexity
- 切回 `model.train()`

```text
perplexity = exp(validation_loss)
```

## 12.6 Checkpoint

保存：

```python
{
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "step": step,
    "epoch": epoch,
    "batch_position": batch_position,
    "config": config,
    "torch_rng_state": ...,
    "cuda_rng_state": ...,
}
```

恢复训练必须恢复：

- 模型
- optimizer
- step
- learning rate 位置
- RNG
- sampler epoch
- 数据位置

## 12.7 Checkpoint 测试

执行两组实验：

```text
实验 A：连续训练 100 steps
实验 B：训练 50 steps -> 保存 -> 恢复 -> 再训练 50 steps
```

在固定 seed 下，两者最终参数和 loss 应当高度接近。

建议 commit：

```text
feat: add resumable mixed-precision trainer
```

---

# 13. 阶段七：Mac 冒烟训练

配置：

```yaml
model: jarvislm-smoke
device: mps
max_seq_len: 256
micro_batch_size: 2
grad_accum_steps: 8
max_steps: 2000
learning_rate: 0.0003
warmup_steps: 100
```

Mac 阶段目标不是获得高质量模型，而是验证：

- [ ] 数据可以持续读取。
- [ ] loss 可以下降。
- [ ] validation loss 可以下降。
- [ ] checkpoint 可以恢复。
- [ ] 训练不会持续增加内存。
- [ ] 模型可以生成文本。
- [ ] 所有实验可以通过配置启动。

Mac 阶段完成后，生成：

```text
artifacts/figures/smoke_train_loss.png
artifacts/figures/smoke_val_loss.png
artifacts/samples/smoke_samples.txt
artifacts/metrics/smoke_summary.json
```

质量门槛：

```text
Q1：单 batch 可过拟合
Q2：小语料 loss 正常下降
Q3：checkpoint 可恢复
Q4：生成流程可运行
```

只有全部通过，才能进入云端训练。

---

# 14. 阶段八：云端迁移

## 14.1 Modal 用途

Modal 用于：

- 100-500 steps 云端冒烟测试
- GPU 兼容性测试
- 显存测试
- 短消融实验
- SFT/评测
- 可恢复任务

Modal 单次任务最长运行时间有限，因此必须将 checkpoint 写入持久化 Volume，并支持重新启动后恢复。

## 14.2 RunPod 用途

RunPod Pod 用于：

- 正式长时间预训练
- 1B-2B tokens 训练
- A100/L40S 连续训练
- 持久化数据和 checkpoint

建议：

```text
开发：Modal A10/L40S
正式训练：RunPod L40S 48GB 或 A100 80GB
在线 Demo：Modal 或 RunPod Serverless
```

## 14.3 云端启动前检查

- [ ] 所有路径来自配置。
- [ ] 不硬编码本地绝对路径。
- [ ] checkpoint 输出到持久化存储。
- [ ] 数据已经预先准备。
- [ ] W&B 日志正常。
- [ ] 训练结束自动退出。
- [ ] 设置支出预算。
- [ ] 设置 Pod 自动停止。
- [ ] checkpoint 已备份到外部存储。

---

# 15. 阶段九：成本与吞吐基准

正式训练前，只运行 200 steps。

记录：

```text
GPU
GPU memory
model parameters
micro batch size
gradient accumulation
sequence length
tokens per step
tokens per second
seconds per step
peak memory
validation loss
```

成本计算：

```text
训练小时
= target_tokens / tokens_per_second / 3600
```

```text
预计 GPU 成本
= 训练小时 × GPU 每小时价格
```

只有当成本可接受时，才启动正式训练。

预算规则：

- [ ] 第一次测试不超过 30 分钟。
- [ ] 第二次测试不超过 2 小时。
- [ ] 正式训练前计算完整预算。
- [ ] 每 100-500 steps 保存 checkpoint。
- [ ] 每次训练设置明确的最大 steps。
- [ ] 训练结束自动关闭实例。

---

# 16. 阶段十：正式预训练 JarvisLM-120M

## 16.1 数据

第一阶段：

```text
FineWeb-Edu 子集
目标：1B tokens
```

推荐拆分：

```text
训练：约 990M tokens
验证：约 10M tokens
```

验证集一旦确定，不再变化。

## 16.2 训练设置

初始建议：

```yaml
model: jarvislm-120m
max_seq_len: 1024
precision: bf16
optimizer: adamw
learning_rate: 0.0003
min_learning_rate: 0.00003
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_ratio: 0.02
grad_clip: 1.0
```

micro batch 和 gradient accumulation 根据显存基准确定。

## 16.3 训练期间检查

每次评估记录：

- training loss
- validation loss
- perplexity
- gradient norm
- learning rate
- tokens/s
- GPU memory
- 随机生成样例

## 16.4 停止条件

满足任一条件停止：

- 达到目标 token budget。
- validation loss 长时间不再改善。
- 出现持续 NaN。
- 训练成本超过预算。
- 数据或实现发现严重问题。

## 16.5 正式训练验收

- [ ] 完成不少于 1B tokens。
- [ ] validation loss 明显低于随机初始化。
- [ ] 训练过程无持续不稳定。
- [ ] 最终 checkpoint 可以独立加载。
- [ ] 生成结果基本符合语料风格。
- [ ] 保存完整配置、commit SHA 和数据 manifest。

---

# 17. 阶段十一：AdamW 与 Muon 实验

## 17.1 实验设计

为了节约成本，先进行短实验：

```text
模型：JarvisLM-120M
训练量：每组 50M-100M tokens
数据：相同
初始化：相同
batch：相同
seed：相同
```

实验矩阵：

| Run | Optimizer | Learning Rate | Token Budget | Seed |
|---|---|---:|---:|---:|
| A1 | AdamW | 搜索值 A | 50M | 42 |
| A2 | AdamW | 搜索值 B | 50M | 42 |
| M1 | Muon + AdamW | 搜索值 A | 50M | 42 |
| M2 | Muon + AdamW | 搜索值 B | 50M | 42 |

找到较优学习率后再运行完整对照。

## 17.2 Muon 参数划分

Muon 处理：

- Attention projection matrices
- MLP 2D matrices

AdamW 处理：

- Embedding
- LM Head
- Norm parameters
- 1D 参数
- bias

必须确保：

```text
Muon 参数数量 + AdamW 参数数量 = 模型总参数数量
```

## 17.3 结果表

| Optimizer | Final Val Loss | PPL | Tokens/s | Peak VRAM | Time | Cost |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | | | | | | |
| Muon | | | | | | |

## 17.4 结论要求

不能只写“Muon 更好”。

必须说明：

- 是否更快收敛
- 是否影响吞吐
- 是否增加显存
- 是否对学习率更敏感
- 是否值得保留
- 结果是否符合假设

---

# 18. 阶段十二：实现 KV Cache

## 18.1 第一版推理

先实现无缓存生成：

```text
每生成一个 token，都重新计算完整上下文。
```

确保生成正确后，再加入 KV Cache。

## 18.2 KV Cache 数据结构

每层保存：

```text
K: [batch, n_kv_heads, cached_seq_len, head_dim]
V: [batch, n_kv_heads, cached_seq_len, head_dim]
```

区分：

- Prefill：一次处理完整 prompt。
- Decode：之后每次只处理一个新 token。

## 18.3 RoPE Offset

缓存生成时，新 token 不能重新从 position 0 开始。

必须使用：

```text
position = cache_length
```

## 18.4 正确性测试

同一模型、同一输入：

```text
naive_logits
cached_logits
```

两者应当在合理误差内一致。

测试：

- [ ] 单 token decode 一致。
- [ ] 多 token decode 一致。
- [ ] batch size > 1。
- [ ] GQA 模式下 cache shape 正确。
- [ ] 不同 prompt 长度正常。
- [ ] 超过 context window 时正确处理。

## 18.5 性能基准

上下文长度：

```text
32
128
256
512
1024
```

每组生成：

```text
128 new tokens
```

结果表：

| Context | Naive tok/s | Cached tok/s | Speedup | Cache VRAM |
|---:|---:|---:|---:|---:|
| 32 | | | | |
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |

---

# 19. 阶段十三：可选 GQA 实验

在 MHA baseline 正确后加入：

```yaml
n_heads: 12
n_kv_heads: 4
```

重点比较：

- KV Cache 大小
- 推理速度
- 训练吞吐
- 参数量
- validation loss

注意：

GQA 会减少参数量，因此不能简单把质量差异完全归因于注意力形式。

报告中必须明确：

```text
这是固定 d_model/layers 的系统对照，而不是严格参数匹配对照。
```

如果需要严格研究质量，应调整 MLP hidden size，让两组总参数量尽量一致。

---

# 20. 阶段十四：模型评测

## 20.1 必须评测

- validation loss
- perplexity
- generation speed
- peak memory
- checkpoint size
- total training tokens
- GPU hours
- estimated cost

## 20.2 可选下游任务

可通过 `lm-evaluation-harness` 测试：

- LAMBADA
- HellaSwag
- PIQA
- ARC-Easy
- Winogrande

120M 模型在这些任务上的分数可能较低，这是正常的。重点是评测流程正确和结果透明。

## 20.3 生成样例

固定 prompts：

```text
The meaning of life is
In a distant galaxy
Machine learning is
The most important scientific discovery
def fibonacci(n):
```

保存：

- 随机初始化结果
- 训练中间结果
- 最终结果
- 不同 temperature 结果

生成样例属于定性证据，不能代替 validation loss 和 benchmark。

---

# 21. 阶段十五：可选后训练

你已有独立 RLHF 项目，因此这个项目不需要把后训练做得过重。

如果希望提供在线聊天 Demo，可以进行轻量 SFT：

- 使用少量高质量 instruction 数据。
- 只训练一个 epoch。
- 只对 assistant response 计算 loss。
- 保留独立 validation set。
- 比较 Base 与 SFT 的指令遵循能力。

不要在第一版重复实现 PPO、DPO、KTO 等已有项目内容。

本项目简历重点保持为：

```text
预训练系统 + 优化器实验 + KV Cache 推理优化
```

---

# 22. 阶段十六：实验记录规范

每次实验必须记录：

```markdown
## Run ID

### Hypothesis

### Git Commit

### Dataset Manifest

### Model Config

### Training Config

### Hardware

### Token Budget

### Runtime

### Cost

### Final Metrics

### Observations

### Failures

### Conclusion

### Next Action
```

建议 Run ID：

```text
2026-08-01_adamw_smoke_v1
2026-08-03_muon_lr3e4_v1
2026-08-10_jarvislm120m_1b
```

禁止使用：

```text
final
final2
really_final
new_test
```

---

# 23. 阶段十七：自动化测试与 CI

GitHub Actions 至少运行：

```bash
ruff check .
pytest -q
```

CI 不运行：

- 大模型训练
- GPU 测试
- 数据下载
- 完整 benchmark

CI 只运行小配置测试。

核心测试清单：

- [ ] RMSNorm
- [ ] SwiGLU
- [ ] RoPE
- [ ] Causal Attention
- [ ] SDPA 一致性
- [ ] Transformer Block
- [ ] GPT forward/backward
- [ ] Weight tying
- [ ] Dataset shift
- [ ] Train/validation 分离
- [ ] Checkpoint resume
- [ ] KV Cache logits 一致性

---

# 24. 阶段十八：发布

## 24.1 GitHub README

README 必须包含：

1. 项目简介
2. 架构图
3. 模型配置
4. 数据说明
5. 训练方法
6. 云端运行方式
7. 实验结果
8. loss 曲线
9. AdamW/Muon 对照
10. KV Cache benchmark
11. 生成样例
12. 成本说明
13. 已知限制
14. 复现命令
15. 模型权重链接

## 24.2 Hugging Face Model Card

包含：

- 模型参数量
- 数据来源
- token 数量
- 训练硬件
- GPU hours
- 训练成本
- 上下文长度
- tokenizer
- benchmark
- 使用示例
- 局限性
- License

## 24.3 发布物

- [ ] GitHub repository
- [ ] Hugging Face checkpoint
- [ ] W&B public report
- [ ] 训练曲线
- [ ] 实验表格
- [ ] KV Cache benchmark
- [ ] 生成样例
- [ ] 可选在线 Demo

---

# 25. 建议时间表

| 周次 | 目标 |
|---|---|
| 第 1 周 | 环境、配置、RMSNorm、SwiGLU、RoPE |
| 第 2 周 | Attention、Block、GPT、单元测试 |
| 第 3 周 | 数据分片、Dataset、单 batch 过拟合 |
| 第 4 周 | Trainer、validation、checkpoint、Mac 训练 |
| 第 5 周 | Modal 测试、RunPod 基准、120M 配置 |
| 第 6 周 | JarvisLM-120M 正式预训练 |
| 第 7 周 | AdamW/Muon 对照实验 |
| 第 8 周 | KV Cache 和推理 benchmark |
| 第 9 周 | 下游评测、结果分析 |
| 第 10 周 | README、模型卡、发布和简历整理 |

时间可以调整，但必须按质量门槛推进，不能跳过测试直接训练大模型。

---

# 26. 项目质量门槛

## Q0：模块正确

- [ ] 所有单元测试通过。
- [ ] Causal Attention 无未来泄漏。
- [ ] GPT forward/backward 正常。

## Q1：可学习

- [ ] 单 batch 可以过拟合。
- [ ] 小语料 loss 可以下降。

## Q2：可恢复

- [ ] checkpoint 可以恢复。
- [ ] 数据位置和 RNG 可以恢复。
- [ ] 恢复后曲线连续。

## Q3：可迁移

- [ ] Mac、Modal、RunPod 使用相同配置系统。
- [ ] 无本地绝对路径。
- [ ] 云端 200 steps 稳定。

## Q4：完成正式训练

- [ ] JarvisLM-120M 完成目标 token budget。
- [ ] 保存最终权重和验证结果。

## Q5：完成独立优化

- [ ] AdamW/Muon 有公平对照。
- [ ] KV Cache 有正确性和性能测试。

## Q6：可公开复现

- [ ] README 完整。
- [ ] 配置公开。
- [ ] 模型权重公开。
- [ ] 结果和失败记录公开。

---

# 27. Definition of Done

只有满足以下条件，项目才算完成：

- [ ] 模型代码由自己独立实现。
- [ ] 核心模块有测试。
- [ ] 单 batch 过拟合成功。
- [ ] Mac 冒烟训练成功。
- [ ] 云端正式训练成功。
- [ ] 完成至少 1B tokens 训练，或诚实记录实际完成量。
- [ ] checkpoint 可以可靠恢复。
- [ ] 有 validation loss 和 perplexity。
- [ ] 有 AdamW/Muon 对照实验。
- [ ] 有 KV Cache 推理 benchmark。
- [ ] 有训练成本和 GPU 时间。
- [ ] GitHub README 完整。
- [ ] 模型权重可以下载。
- [ ] 简历中的每个数字都有实验记录支持。

---

# 28. 最终简历模板

完成项目后填写真实数字：

```latex
% ---------- Project ----------

\entry
  {Small LLM from Scratch：小型语言模型预训练与推理优化}
  {\projectlink{https://github.com/JarvisZhang24/small-llm-from-scratch}}

\sub{Python、PyTorch、CUDA、tiktoken、DDP、W\&B、Modal、RunPod}

\begin{itemize}[topsep=2pt, itemsep=1pt]
  \item
  使用 PyTorch 从零实现并训练约 \textbf{[X]M 参数}
  Decoder-only 语言模型，集成 RoPE、RMSNorm、SwiGLU
  与因果注意力，构建数据分片、混合精度训练和断点续训流程。

  \item
  在固定 \textbf{[X]B tokens} 训练预算下完成
  AdamW/Muon 对照实验，并实现 KV Cache 推理优化，
  将生成吞吐提升 \textbf{[X] 倍}，最终验证困惑度达到
  \textbf{[X]}。
\end{itemize}
```

在实验完成前不要填写虚构数字。

---

# 29. 现在立即执行的任务

第一天只完成以下内容：

- [ ] 初始化 Git。
- [ ] 创建虚拟环境。
- [ ] 创建目录结构。
- [ ] 编写 `.gitignore`。
- [ ] 编写 `ModelConfig`。
- [ ] 实现 RMSNorm。
- [ ] 编写 RMSNorm 测试。
- [ ] 运行 `pytest`。
- [ ] 提交第一个功能 commit。

第一天完成标准：

```bash
pytest -q
```

输出全部通过，并存在以下 commits：

```text
chore: initialize project structure
feat: add model configuration
feat: implement rmsnorm with tests
```

完成后再进入 SwiGLU 和 RoPE，不要提前开始下载大规模数据或租用 GPU。
