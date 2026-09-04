# 从零理解 Tiny GPT

这是一个教学用字符级 GPT。目标不是训练出实用聊天机器人，而是把现代语言模型最核心的闭环完整跑一遍：

`文字 → token → embedding → Transformer → logits → loss → 反向传播 → 参数更新 → checkpoint → 生成`

项目刻意只依赖 PyTorch，不使用 Hugging Face，也不隐藏训练循环。

第一次学习请从 [`docs/README.md`](docs/README.md) 开始，按课程顺序阅读和实验。

## 文件地图

- `tinygpt.py`：tokenizer、因果自注意力、Transformer block、GPT 和生成算法
- `train.py`：数据切分、batch、损失评估、反向传播、优化器、checkpoint 与断点续训
- `generate.py`：加载 checkpoint 并逐 token 生成
- `inspect_forward.py`：显示一次前向传播的张量形状与注意力矩阵
- `test_tinygpt.py`：tokenizer、形状、因果遮罩与生成的最小单元测试
- `data/tiny_chinese.txt`：教学用中文小语料
- `docs/`：从零学习 Tiny GPT 的完整课程、术语表和开源清单

## 运行顺序

创建环境并安装 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

先观察一次前向传播：

```bash
.venv/bin/python inspect_forward.py
```

做一次快速训练验证：

```bash
.venv/bin/python train.py --steps 30 --eval-interval 10 --eval-iters 3 \
  --n-layer 2 --n-embd 64 --block-size 64 --batch-size 8
```

运行测试：

```bash
.venv/bin/python -m unittest -v
```

运行默认教学训练（约 13.5 万参数、1,200 步）：

```bash
.venv/bin/python train.py
```

先保存一个新版完整训练 checkpoint，再恢复并新增 20 个训练 step：

```bash
.venv/bin/python train.py \
  --steps 30 \
  --output checkpoints/demo-step-30.pt

.venv/bin/python train.py \
  --resume checkpoints/demo-step-30.pt \
  --steps 20 \
  --output checkpoints/demo-step-50.pt
```

`--steps` 在恢复模式下表示本次新增步数；新 checkpoint 会累计原有 `training_steps`。`--output` 必须与 `--resume` 使用不同路径，避免覆盖唯一恢复点。旧版缺少 optimizer/RNG 的推理兼容 checkpoint 仍能生成，但不能完整恢复训练。

加载模型并生成：

```bash
.venv/bin/python generate.py --prompt "语言模型" --tokens 200
```

## 建议的阅读顺序

1. 阅读 `CharTokenizer`，理解文字为什么要变成整数。
2. 阅读 `make_batch`，手动核对输入 `x` 和右移后的目标 `y`。
3. 阅读 `TinyGPT.forward`，跟踪 `(B, T)` 如何变成 `(B, T, V)`。
4. 阅读 `CausalSelfAttention`，重点看 Q、K、V 的形状和 causal mask。
5. 阅读训练循环中的 `loss.backward()` 与 `optimizer.step()`。
6. 阅读 `generate`，理解训练和推理的区别。

## 重要边界

小语料很容易过拟合，生成质量有限，这是预期现象。这个项目用来理解机制；后续会在此基础上进入标准 tokenizer、预训练模型和 LoRA 微调。

`--device auto` 会优先使用 Apple MPS，不可用时自动退回 CPU。当前默认小模型在 CPU 上也能很快完成训练。

## 开源许可证

本项目采用 [MIT License](LICENSE)。
