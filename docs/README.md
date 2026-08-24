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
| 第 2 课 | Embedding、位置编码、BTC/BSH/BNSD | 下一课 |
| 第 3 课 | Q、K、V 与因果 Self-Attention | 待学习 |
| 第 4 课 | 多头注意力、合并 Head 与输出投影 | 待学习 |
| 第 5 课 | Transformer Block、残差连接、LayerNorm、FFN | 待学习 |
| 第 6 课 | Logits、Softmax 与交叉熵 | 待学习 |
| 第 7 课 | 梯度、反向传播与 AdamW | 待学习 |
| 第 8 课 | 训练循环、验证集与过拟合 | 待学习 |
| 第 9 课 | 自回归生成、temperature 与 top-k | 待学习 |
| 第 10 课 | Checkpoint、加载与继续训练 | 待学习 |
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

下一步是理解 `(B,T)` 如何经 embedding 变成 `(B,T,C)`，以及 token 信息为什么还需要加上位置信息。
