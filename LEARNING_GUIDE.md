# Tiny GPT 学习指南

不要急着背公式。先把一条文本从输入跟踪到输出，再回头理解每一步为什么存在。

## 1. 先运行，不训练

```bash
.venv/bin/python inspect_forward.py
```

重点观察四个形状：

- token IDs：`(B, T)`，B 是批次数量，T 是序列长度。
- embedding：概念上是 `(B, T, C)`，C 是向量维度。
- attention：`(B, H, T, T)`，H 是注意力头数。
- logits：`(B, T, V)`，V 是词表大小。

注意力矩阵右上角全为零，这就是因果遮罩。第 t 行不能读取 t 右侧的未来字符。

阅读顺序：

1. `CharTokenizer.encode`
2. `TinyGPT.forward`
3. `CausalSelfAttention.forward`
4. `FeedForward.forward`

## 2. 理解训练数据

在 `train.py` 找到 `make_batch`。如果一段文本是：

```text
语言模型
```

输入和目标相当于：

```text
输入：语 言 模
目标：言 模 型
```

模型在三个位置同时练习预测下一个字符。这就是自回归语言建模。

## 3. 跟踪一次参数更新

训练循环中最关键的五行是：

```python
_, loss = model(inputs, targets)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

它们依次完成前向传播、清空旧梯度、反向传播、限制过大梯度和更新参数。

运行 30 步小实验：

```bash
.venv/bin/python train.py --steps 30 --eval-interval 10 --eval-iters 3
```

比较 step 0 和 step 30 的 loss。随机模型的 loss 通常接近 `ln(词表大小)`；本项目词表约 490，所以初始 loss 约为 6.2。

## 4. 看懂一次完整训练

默认配置：

- 135,040 个可训练参数
- 2 层 Transformer
- 2 个注意力头
- embedding 维度 64
- 上下文窗口 64 个字符
- 训练 1,200 步

本机已经完成过一次训练，主要结果如下：

| step | train loss | val loss |
| ---: | ---: | ---: |
| 0 | 6.2004 | 6.2006 |
| 300 | 3.2948 | 4.2796 |
| 600 | 1.9843 | 3.9334 |
| 900 | 1.2620 | 4.0740 |
| 1200 | 0.7993 | 4.1572 |

训练损失持续下降，但验证损失在约 600 步后不再改善。这不是程序失败，而是小数据上的典型过拟合：模型越来越会记忆训练部分，却没有继续提升对验证部分的预测。

## 5. 理解生成

```bash
.venv/bin/python generate.py --prompt "语言模型" --tokens 200
```

`TinyGPT.generate` 每轮只做四件事：

1. 截取上下文窗口内的 token。
2. 前向传播，取最后一个位置的 logits。
3. 用 temperature 和 top-k 调整采样范围。
4. 采样下一个 token，并拼回序列。

生成文本像训练语料但不完全通顺，是因为模型和数据都非常小。这里评价的是机制是否跑通，而不是聊天质量。

## 6. 建议亲手修改的实验

每次只改一个变量，并记录训练和验证 loss：

1. 把 `--steps` 改为 100、600、1200，比较欠拟合与过拟合。
2. 把 `--block-size` 从 64 改成 16，观察较短上下文的影响。
3. 生成时分别使用 `--temperature 0.2` 和 `1.2`。
4. 暂时移除 causal mask，观察训练指标，并思考为什么生成阶段仍会失败。
5. 把 `--n-layer` 从 2 改成 4，比较参数量、速度和损失。

完成这些实验后，再进入子词 tokenizer、预训练模型和 LoRA，会更容易理解它们解决了什么问题。
