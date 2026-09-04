# Tiny GPT 学习路线

这套教程以“亲手跑通一个模型”为主线。先建立直觉，再阅读代码，最后补充公式和现代 LLM 的工程实现。

完整数据流：

```text
文字
  → token IDs
  → token/position embedding
  → 多头因果自注意力
  → 前馈网络
  → logits
  → 交叉熵 loss
  → 反向传播与参数更新
  → checkpoint
  → 自回归生成
```

## 课程目录

| 课程 | 主题 | 状态 |
| --- | --- | --- |
| [第 1 课](lessons/01-tokenizer-and-batches.md) | Tokenizer、`self`、训练样本、`torch.stack`、`make_batch` | 已完成 |
| [第 2 课](lessons/02-embeddings-and-shapes.md) | Embedding、位置编码、BTC/BSH/BNSD | 已完成 |
| [第 3 课](lessons/03-qkv-and-self-attention.md) | Q、K、V 与因果 Self-Attention | 已完成 |
| [第 4 课](lessons/04-multi-head-attention.md) | 多头注意力、合并 Head 与输出投影 | 已完成 |
| [第 5 课](lessons/05-transformer-block.md) | Transformer Block、残差连接、LayerNorm、FFN | 已完成 |
| [第 6 课](lessons/06-logits-softmax-cross-entropy.md) | Logits、Softmax 与交叉熵 | 已完成 |
| [第 7 课](lessons/07-gradients-backprop-adamw.md) | 梯度、反向传播与 AdamW | 已完成 |
| [第 8 课](lessons/08-training-validation-overfitting.md) | 训练循环、验证集与过拟合 | 已完成 |
| [第 9 课](lessons/09-autoregressive-generation.md) | 自回归生成、temperature 与 top-k | 已完成 |
| [第 10 课](lessons/10-checkpoint-loading-and-resume.md) | Checkpoint、加载与继续训练 | 已完成 |
| 第 11 课 | 从 Tiny GPT 对照现代 LLM | 待学习 |
| 第 12 课 | 子词 tokenizer、预训练模型与 LoRA | 待学习 |

## 每课怎么学

每一课遵循同样的节奏：

1. 明确本课只解决哪个问题。
2. 用极小例子手算一次。
3. 对照项目中的真实代码。
4. 跟踪输入和输出 shape。
5. 运行实验验证直觉。
6. 回答验收题，不依赖背诵。

不要同时修改多个超参数。实验时一次只改变一个变量，并记录现象和解释。

## 配套资料

- [术语与形状速查表](GLOSSARY.md)
- [完整训练与生成实验](../LEARNING_GUIDE.md)
- [GitHub 开源前检查清单](OPEN_SOURCE_CHECKLIST.md)
- [核心模型代码](../tinygpt.py)
- [训练代码](../train.py)

## 当前进度

已经掌握：

- 字符如何编码为 token ID，又如何解码回文字。
- `self` 为什么表示当前 Python 对象。
- 如何从长语料随机抽取 `(B,T)` 训练 batch。
- 输入和目标为什么错开一个 token。
- `(B,T)` 如何经 embedding 变成 `(B,T,C)`。
- BTC、BSH 与 BNSD 只是不同的维度命名约定。
- 隐藏维度满足 `C=N×D`，并能在 BSH 与 BNSD 之间转换。
- `(B,T,V)` 包含 `B×T` 份 next-token prediction，每份预测有 `V` 个候选分数。
- 生成时只取最后位置 logits，并能区分整数索引 `-1` 与切片 `-1:` 对 shape 的影响。
- temperature 调整概率差距，top-k 限制候选池，multinomial 根据概率抽取 token ID。
- prompt、`max_new_tokens`、context window 与最终输出长度之间的关系。
- 为什么当前实现逐 token 自回归生成，并且尚未使用 KV Cache 和 EOS 提前停止。

- checkpoint、`state_dict`、Parameter 与 Buffer 分别表示什么。
- 推理加载、warm start 与完整 resume 的区别。
- 为什么完整恢复需要模型、AdamW、累计 step、tokenizer/config 和 RNG 状态。
- 验证过程如何临时使用并恢复 RNG，避免改变后续训练 batch。
- `map_location="cpu"`、`weights_only=True` 和严格权重匹配分别解决什么问题。
- missing key、unexpected key、size mismatch 和 optimizer 设备不匹配的区别。

下一课将把这个 Tiny GPT 的结构逐项映射到现代 LLM。
