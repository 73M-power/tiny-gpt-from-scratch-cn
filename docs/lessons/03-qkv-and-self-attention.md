# 第 3 课：Q、K、V 与 Self-Attention

> 状态：已完成。本课覆盖 Q/K/V、缩放点积、因果遮罩、softmax 和 Value 加权的完整闭环。

## 学习目标

完成本课后，应该能够解释：

1. 为什么同一个输入 `x` 要分别生成 Q、K、V。
2. `nn.Linear(C,C)` 如何保持 shape、改变表示。
3. `Q @ Kᵀ` 为什么得到 `(B,N,T,T)`。
4. attention score 的行、列分别表示什么。
5. 为什么分数需要除以 `sqrt(D)`。
6. softmax 后的权重如何对 V 加权求和。

## 0. 前置 Shape 检查

给定：

```text
B=3, T=10, C=96, N=4
```

则：

```text
D = C/N = 24
Q shape = (B,N,T,D) = (3,4,10,24)
Q @ Kᵀ = (B,N,T,T) = (3,4,10,10)
```

## 1. Self-Attention 要解决什么问题

Embedding 阶段已经告诉模型每个位置“是什么 token、位于哪里”，但一个位置还没有读取其他位置的信息。

Self-Attention 要让每个位置回答：

```text
为了更新我自己的表示，序列中哪些位置与我有关？
我应该从它们那里读取多少信息？
```

例如处理“它”时，模型可能需要读取前文中的名词；处理“模型”时，可能需要关注“训练”“参数”等上下文。具体关系不是人工编写规则，而是从训练目标中学习。

## 2. Q、K、V 的三个角色

可以先使用“检索”直觉：

| 向量 | 直觉问题 | 作用 |
| --- | --- | --- |
| Q，Query | 我在寻找什么？ | 当前位置发出的查询 |
| K，Key | 我能以什么特征被找到？ | 每个位置用于匹配的索引 |
| V，Value | 如果找到我，我提供什么？ | 真正传递给查询位置的信息 |

重要区别：

```text
Q 和 K 决定“关注多少”
V 决定“读取什么内容”
```

Q、K、V 都来自同一个输入 `x`，但使用不同的可训练线性投影，所以能够承担不同角色。

## 3. 三个线性投影

项目初始化代码：

```python
self.query = nn.Linear(C, C, bias=False)
self.key = nn.Linear(C, C, bias=False)
self.value = nn.Linear(C, C, bias=False)
```

数学表示：

```text
Q = XWq
K = XWk
V = XWv
```

`Wq`、`Wk`、`Wv` 是三套不同的可训练参数。当前 `C=64` 时，每个权重矩阵都是 `(64,64)`。

线性层只作用于最后一维：

```text
X              (B,T,C)
Wq             (C,C)
Q = XWq        (B,T,C)
```

因此 shape 没有改变，但每个 token 的 64 维表示被重新组合。保持相同 shape 不代表数值或含义没有变化。

## 4. 拆分多头

项目代码：

```python
q = self.query(x).view(B, T, N, D).transpose(1, 2)
k = self.key(x).view(B, T, N, D).transpose(1, 2)
v = self.value(x).view(B, T, N, D).transpose(1, 2)
```

shape：

```text
(B,T,C)
→ view      (B,T,N,D)
→ transpose (B,N,T,D)
```

每个 head 都能看到全部 T 个位置，只是使用自己对应的 D 维子空间计算关系。

## 5. `Q @ Kᵀ` 计算匹配分数

单个 batch、单个 head 暂时忽略 B 和 N：

```text
Q  shape = (T,D)
K  shape = (T,D)
Kᵀ shape = (D,T)
```

矩阵乘法：

```text
(T,D) @ (D,T) = (T,T)
```

保留 batch 和 head 后：

```text
(B,N,T,D) @ (B,N,D,T) = (B,N,T,T)
```

项目代码：

```python
scores = q @ k.transpose(-2, -1)
```

最后两个维度执行矩阵乘法，前面的 B 和 N 是并行维度。

## 6. Score 矩阵的行与列

对于一个 head：

```text
score[i,j] = dot(Q[i], K[j])
```

- 第 `i` 行：位置 i 发出的 query。
- 第 `j` 列：位置 j 提供的 key。
- `score[i,j]`：位置 i 对位置 j 的原始匹配程度。

例如 `T=4`：

```text
             被读取的 key 位置 j
             0     1     2     3
query i=0  [ ·     ·     ·     · ]
query i=1  [ ·     ·     ·     · ]
query i=2  [ ·     ·     ·     · ]
query i=3  [ ·     ·     ·     · ]
```

每一行回答：当前位置应该关注所有位置各多少。此时它还是原始分数，可以为负数，也不要求一行之和等于 1。

## 7. 为什么除以 `sqrt(D)`

项目实际代码：

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
```

当 D 增大时，D 个乘积相加得到的点积数值通常会变大。如果直接送入 softmax，分布容易过早变得极端，使梯度变小、训练不稳定。

除以 `sqrt(D)` 可以让不同 head dimension 下的分数尺度更稳定。这就是 scaled dot-product attention 中的 “scaled”。

## 8. 因果遮罩：只能读取过去和自己

GPT 训练 next-token prediction 时不能偷看未来。Query 位于索引 `i` 时，只允许关注满足 `j <= i` 的 Key：

```text
              Key 索引
              0  1  2  3
Query 索引 0  ✓  ✗  ✗  ✗
Query 索引 1  ✓  ✓  ✗  ✗
Query 索引 2  ✓  ✓  ✓  ✗
Query 索引 3  ✓  ✓  ✓  ✓
```

注意对角线允许保留：索引 4 可以关注 `0、1、2、3、4`，不仅是它之前的位置。

项目代码先创建下三角矩阵：

```python
mask = torch.tril(torch.ones(block_size, block_size))
```

再把未来位置填为负无穷：

```python
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
```

例如 `score[2,5]` 在 raw scores 中表示索引 2 的 Query 与索引 5 的 Key 的匹配分数；由于 `5>2`，遮罩后它变为 `-inf`，softmax 后权重严格为 0。

完整 `T×T` 分数矩阵仍然一次计算，再统一遮罩，这是适合张量并行的实现方式。

## 9. Softmax：把每行分数变成权重

```python
weights = F.softmax(scores, dim=-1)
```

scores 的 shape 可以写成：

```text
(B,N,T_query,T_key)
```

`dim=-1` 沿最后的 Key 位置维归一化。对每个 batch、head 和 Query：

```text
weights[b,n,i,:].sum() = 1
```

softmax 后每个值位于 0 到 1；被填成 `-inf` 的未来位置因为 `exp(-inf)=0`，最终权重为 0。

例如：

```text
masked scores = [2.0, 1.0, -inf]
weights       ≈ [0.731, 0.269, 0.000]
```

## 10. `weights @ V`：真正读取内容

```text
weights (B,N,T,T)
V       (B,N,T,D)
结果     (B,N,T,D)
```

位置 `i` 的输出是所有 Value 的加权和：

```text
output[i] = Σ_j weights[i,j] × V[j]
```

例如：

```text
weights = [0.25, 0.75]
V[0]    = [1,0]
V[1]    = [0,2]

output  = 0.25×[1,0] + 0.75×[0,2]
        = [0.25,1.50]
```

因此：

```text
Q/K：决定读取哪些位置以及读取比例
V：提供被读取的实际向量内容
```

可以类比为：Q 是搜索词，K 是资料标签，Q/K 匹配产生资料权重，V 是资料正文。

## 11. 合并多个 Head

完成 Value 加权后：

```text
attended.shape = (B,N,T,D)
```

项目代码：

```python
attended = attended.transpose(1, 2).contiguous().view(B, T, C)
```

shape 变化：

```text
(B,N,T,D)
→ transpose (B,T,N,D)
→ merge N×D (B,T,C)
```

因为 `C=N×D`，Attention 的输入和合并后的输出都可以保持 `(B,T,C)`。

练习实例：

```text
B=3, N=4, T=10, D=24, C=96
weights @ V 结果 = (3,4,10,24)
合并 Head 后     = (3,10,96)
```

另一个实例：

```text
weights = (2,3,8,8)
V       = (2,3,8,16)
结果     = (2,3,8,16)
合并后 C=3×16=48，所以输出 = (2,8,48)
```

## 12. 完整 Attention 闭环

```text
X (B,T,C)
→ Q/K/V (B,N,T,D)
→ QKᵀ/√D 得到 scores (B,N,T,T)
→ causal mask 将未来分数设为 -inf
→ softmax 得到每行和为 1 的 weights
→ weights @ V 得到 (B,N,T,D)
→ 合并 Head 回到 (B,T,C)
```

## 验收题

1. Q 和 K 决定什么，V 决定什么？
2. `nn.Linear(C,C)` 为什么 shape 不变，但表示已经变化？
3. 在 raw score 矩阵中，`score[2,5]` 表示谁匹配谁？为什么最终权重为 0？
4. `Q @ Kᵀ` 的结果为什么是 `(T,T)`，而不是 `(D,D)`？
5. `softmax(dim=-1)` 归一化的是哪个维度？
6. 为什么 `weights @ V` 后仍然保留每个 Query 的输出位置？
7. 怎样从 `(B,N,T,D)` 合并回 `(B,T,C)`？

完成后进入第 4 课：不同 Head 如何学习不同关系、为什么需要输出投影，以及 attention dropout 与 residual dropout 位于哪里。
