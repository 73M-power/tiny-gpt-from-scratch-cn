# 第 2 课：Embedding 与张量形状

## 学习目标

完成本课后，应该能够解释：

1. 为什么 token ID 还不能直接表示语义。
2. `nn.Embedding(V,C)` 本质上如何查表。
3. `(B,T)` 为什么会变成 `(B,T,C)`。
4. token embedding 和 position embedding 各自提供什么信息。
5. BTC、BSH、BNSD 等 shape 记法之间的关系。

## 1. Embedding 是可训练查找表

Tokenizer 产生的 ID 只是类别编号。ID 的大小和距离没有语言含义，所以需要把每个 ID 映射为一个可学习向量。

项目代码：

```python
self.token_embedding = nn.Embedding(
    config.vocab_size,
    config.n_embd,
)
```

当前配置为：

```text
V = vocab_size = 490
C = n_embd     = 64
```

因此参数矩阵为：

```text
token_embedding.weight.shape = (V,C) = (490,64)
```

输入 token ID `427` 时，embedding 会取参数矩阵的第 427 行，返回长度为 64 的向量。它等价于查表，也可以在数学上看作 one-hot 向量与 embedding 矩阵相乘，但实际实现不需要生成巨大的 one-hot 矩阵。

## 2. C 维从哪里出现

输入只有 token 编号：

```text
token_ids.shape = (B,T)
```

每个编号查出一个 C 维向量：

```text
(B,T) → embedding lookup → (B,T,C)
```

真实例子：

```text
(1,12) → (1,12,64)
```

其中：

- `B`：batch size。
- `T`：time/sequence length。
- `C`：channel/embedding/hidden dimension。

这 64 个维度没有预先规定的单独语义。它们随机初始化，通过 loss、反向传播和优化器共同学成对预测有用的表示。

## 3. 为什么需要位置信息

“我爱你”和“你爱我”包含相同 token，但顺序和意义不同。token embedding 只说明 token 是什么，不能单独说明它位于哪里。

本项目使用可学习的绝对位置 embedding：

```python
self.position_embedding = nn.Embedding(
    config.block_size,
    config.n_embd,
)
```

当前参数矩阵：

```text
(block_size,C) = (64,64)
```

长度为 `T` 的输入创建位置编号：

```python
positions = torch.arange(time, device=token_ids.device)
```

当 `T=12` 时：

```text
positions = [0,1,2,3,4,5,6,7,8,9,10,11]
positions.shape = (T,) = (12,)
```

位置查表后：

```text
position_embedding(positions).shape = (T,C) = (12,64)
```

## 4. Token 与位置向量相加

核心代码：

```python
x = self.token_embedding(token_ids) + self.position_embedding(positions)
```

shape：

```text
token embedding    : (B,T,C)
position embedding :   (T,C)
```

PyTorch 广播会把 `(T,C)` 视为 `(1,T,C)`，并沿 batch 维复用，所以结果仍为：

```text
x.shape = (B,T,C)
```

对于位置 `t` 的 token：

```text
x[b,t] = token_vector[token_id] + position_vector[t]
```

同一 token 出现在不同位置时，token vector 相同、position vector 不同，因此最终表示不同。

## 5. BTC、BSH 与 BNSD

不同项目经常为相同维度使用不同字母：

| 本项目 | 常见写法 | 含义 |
| --- | --- | --- |
| B | B | batch size |
| T | S | time/sequence length |
| C | H | channel/hidden size |

因此：

```text
BTC ≈ BSH
```

进入多头注意力后，隐藏维度 `C` 会被拆成：

```text
C = N × D
```

其中：

- `N`：注意力头数。
- `D`：每个头的维度。

当前配置：

```text
C = 64
N = 2
D = 32
```

形状变化：

```text
(B,T,C)
→ view (B,T,N,D)
→ transpose (B,N,T,D)
```

这就是常见的 BNSD，只是本项目把 sequence 写作 `T`：

```text
BNTD ≈ BNSD
```

## 6. 后续注意力的完整 shape 预览

当前示例：

```text
B=1, T=12, C=64, N=2, D=32, V=490
```

```text
token IDs       (B,T)     = (1,12)
embedding       (B,T,C)   = (1,12,64)    ≈ BSH
Q/K/V           (B,N,T,D) = (1,2,12,32)  ≈ BNSD
attention score (B,N,T,T) = (1,2,12,12)  ≈ BNSS
merged output   (B,T,C)   = (1,12,64)    ≈ BSH
logits          (B,T,V)   = (1,12,490)
```

最重要的转换是：

```text
BTC → BNTD → BTC
BSH → BNSD → BSH
```

注意：不同资料可能用 `H` 表示 hidden size，也可能表示 head 数量。不要只凭字母猜含义，应先查看上下文或代码中的维度定义。

## 7. 参数量

当前模型中：

```text
token embedding 参数   = 490 × 64 = 31,360
position embedding 参数 = 64 × 64  = 4,096
```

它们都包含在 `model.parameters()` 中，会随训练更新。

## 8. 与现代 LLM 的对应

本项目使用可学习的绝对位置 embedding，优点是直观。许多现代 LLM 使用 RoPE（旋转位置编码），实现不同，但都在解决顺序和相对位置问题。

## 9. 常见理解校准

- 每个 token 在当前字符 tokenizer 中对应一个 ID；一段文本则对应一串 ID。
- C 维可以先直观理解为“从多个角度表示 token”，但它不是人为定义的 64 项评分。每一维的含义由训练共同决定，通常不能单独命名。
- 相同 token 的初始 token embedding 相同；位置 embedding 只补充它出现在哪里。结合上下文后的语义差异，要到 Self-Attention 处理后才产生。
- `N` 是注意力头数，`D` 是每个头的维度，并满足 `C=N×D`。

例如 `B=4,T=16,C=64,N=2`：

```text
x                 (B,T,C)   = (4,16,64)
Q/K/V 线性投影后   (B,T,C)   = (4,16,64)
拆分多头           (B,T,N,D) = (4,16,2,32)
交换维度           (B,N,T,D) = (4,2,16,32)
attention scores  (B,N,T,T) = (4,2,16,16)
合并多头           (B,T,C)   = (4,16,64)
```

## Shape 练习

给定：

```text
B=4, T=16, C=64, N=2
```

请计算：

1. `token_ids` 的 shape。
2. token embedding 的 shape。
3. `positions` 的 shape。
4. position embedding 的 shape。
5. 两种 embedding 相加后 `x` 的 shape。
6. 每个注意力头的维度 `D`。
7. 拆分多头后 Q 的 BNSD/BNTD shape。

## 验收标准

能够用自己的话解释以下句子：

> `(B,T)` 是一批 token 编号；embedding 为每个编号查出 C 维向量，得到 `(B,T,C)`；多头注意力再把 `C` 拆成 `N×D`，得到 `(B,N,T,D)`。

完成后进入第 3 课：Q、K、V 与单头 Self-Attention。
