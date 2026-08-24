# 第 4 课：多头注意力、合并 Head 与输出投影

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. 多个 Head 为什么不是把 token 分组处理。
2. `C=N×D` 如何让总隐藏维度保持不变。
3. 为什么每个 Head 可以产生不同的注意力矩阵。
4. 怎样从 `(B,N,T,D)` 合并回 `(B,T,C)`。
5. 为什么合并后还需要输出投影 `proj`。
6. Attention Dropout 与 Residual Dropout 分别作用在哪里。

## 1. 多头注意力多在哪里

单头注意力对每个 Query 只产生一套长度为 T 的注意力权重。多头注意力并行产生 N 套权重：

```text
Head 0：一套 T×T attention map
Head 1：另一套 T×T attention map
...
Head N-1：第 N 套 T×T attention map
```

所有 Head 都能看到完整的 T 个 token。它们不是把序列切成 N 段，而是把每个 token 的 C 维特征拆成 N 个 D 维子空间：

```text
C = N × D
```

当前教学模型：

```text
C=64, N=2, D=32
```

每个 Head 使用自己的 Q/K/V 子空间，因此能学习不同的匹配关系。但“某个 Head 一定负责语法”并非人工规定，也不保证每个 Head 都能被人类清楚解释。

## 2. 本项目怎样生成多个 Head

代码先做一次组合线性投影：

```python
q = self.query(x)
k = self.key(x)
v = self.value(x)
```

每个结果仍为 `(B,T,C)`，然后拆分：

```python
q = q.view(B, T, N, D).transpose(1, 2)
```

得到：

```text
(B,T,C) → (B,T,N,D) → (B,N,T,D)
```

这相当于把投影结果最后的 C 个数按 Head 切成 N 组，并不是让每个 Head 只接收部分 token。

## 3. 每个 Head 独立计算 Attention

```text
Q/K/V      (B,N,T,D)
scores     (B,N,T,T)
weights    (B,N,T,T)
attended   (B,N,T,D)
```

B 和 N 是批量维度，因此每个 batch 中的每个 Head 都独立执行：

```text
softmax(Q_head @ K_headᵀ / sqrt(D)) @ V_head
```

不同 Head 的 Q/K 表示不同，所以即便输入 token 相同，它们得到的 attention map 也可以不同。

## 4. 合并 Head

各 Head 输出：

```text
attended.shape = (B,N,T,D)
```

先交换 Head 与序列维：

```python
attended = attended.transpose(1, 2)
```

得到：

```text
(B,T,N,D)
```

再合并最后两维：

```python
attended = attended.contiguous().view(B, T, C)
```

由于 `C=N×D`：

```text
(B,T,N,D) → (B,T,C)
```

例如：

```text
(3,4,10,24)
→ transpose (3,10,4,24)
→ merge     (3,10,96)
```

## 5. 为什么需要 `contiguous()`

`transpose` 通常只改变 tensor 的维度解释和 stride，不一定重新排列底层内存。`view` 要求目标布局能按连续内存解释，因此代码先调用：

```python
attended.contiguous()
```

它生成符合当前维度顺序的连续内存布局，之后才能安全地把 `N` 与 `D` 合并为 `C`。

对初学阶段可以记成：

```text
transpose 改逻辑顺序
contiguous 整理物理布局
view 合并维度
```

## 6. 输出投影 `proj`

仅仅合并 Head，结果只是把多个 Head 的输出并排拼接：

```text
[Head 0 的 D 维 | Head 1 的 D 维 | ...]
```

项目接着执行：

```python
self.proj = nn.Linear(C, C, bias=False)
output = self.proj(attended)
```

`proj` 允许每个输出维度同时组合来自不同 Head 的信息。它保持 `(B,T,C)` shape 不变，但重新混合多个 Head 的特征。

Attention 最终必须输出 `(B,T,C)`，因为下一步需要与原输入做残差相加：

```python
x = x + attention(...)
```

相加两边必须拥有相同 shape。

## 7. 两种 Dropout

本项目包含：

```python
self.attn_dropout = nn.Dropout(dropout)
self.resid_dropout = nn.Dropout(dropout)
```

Attention Dropout 作用于 softmax 后的连接权重：

```python
weights = self.attn_dropout(weights)
```

它在训练时随机关闭部分注意力连接，降低模型过度依赖某一连接的风险。

Residual Dropout 作用于输出投影之后：

```python
return self.resid_dropout(self.proj(attended))
```

它随机关闭部分输出特征，再交给残差连接。

`model.train()` 时 Dropout 生效；`model.eval()` 时关闭。Dropout 没有可训练参数。训练模式中保留下来的值会按比例缩放，因此经过 Attention Dropout 后，一行权重不一定仍严格加和为 1。

## 8. 参数量

本实现有四个无 bias 的 `C→C` 线性层：

```text
Wq、Wk、Wv、Wo(proj)
```

Attention 线性参数量：

```text
4 × C²
```

当前 `C=64`：

```text
4 × 64² = 16,384
```

拆成更多 Head 不会改变这些组合投影的总参数量，只会改变 `N` 和 `D` 的划分，前提是 C 保持不变。

## 9. 完整 Shape

给定：

```text
B=3, T=10, C=96, N=4, D=24
```

```text
X                  (3,10,96)
Q/K/V              (3,4,10,24)
scores/weights     (3,4,10,10)
各 Head 输出        (3,4,10,24)
transpose          (3,10,4,24)
merge              (3,10,96)
output projection  (3,10,96)
```

## 10. 常见理解校准

### Head 看完整序列，但使用特征子空间

每个 Head 都处理全部 T 个 token 位置，所以不是把序列分段。但当 `C=N×D` 时，每个 Head 确实使用投影结果中的一个 D 维特征子空间：

```text
不是：Head 0 只看前几个 token
而是：Head 0 看所有 token 的第 0 组 D 维投影特征
```

### `proj` 是学习混合，不是随机打乱

拼接结果仍按 Head 分区排列。`proj` 使用训练学到的权重，对所有 C 维做加权组合，使一个输出维度可以同时利用多个 Head。它不是随机洗牌，训练完成后同一输入和参数会执行确定的线性变换。

### Dropout 不保证输出正确

Dropout 通过训练时随机关闭部分连接或特征，减少模型过度依赖特定路径的风险。它是一种正则化手段，目标是提高未见数据上的泛化能力，但不保证单次结果一定更正确。

### 固定 C 时，增加 Head 不增加投影参数量

本实现的 Q/K/V/Proj 总参数量始终为：

```text
4×C²
```

当 C 固定时，N 增大意味着 D 减小。例如 `C=128,N=8` 时 `D=16`。同时要求 C 能被 N 整除。

## 阶段结论

```text
所有 Head 都读取完整序列
每个 Head 在 D 维子空间产生独立 attention map
各 Head 输出从 (B,N,T,D) 转为 (B,T,N,D)
合并 N×D 得到 (B,T,C)
proj 以可学习方式混合各 Head
Dropout 只在训练模式中提供正则化
```

## 验收题

1. N 个 Head 是否分别处理不同的 token 片段？
2. `C=128,N=8` 时 D 是多少？
3. 为什么不同 Head 能得到不同的 attention map？
4. `(2,8,12,16)` 合并 Head 后是什么 shape？
5. 为什么 `transpose` 后通常要调用 `contiguous()` 再 `view`？
6. 已经拼接为 `(B,T,C)` 后，为什么还要执行 `proj`？
7. Attention Dropout 和 Residual Dropout 分别作用于哪里？

完成后进入第 5 课：Attention 如何与 LayerNorm、残差连接和前馈网络组成完整 Transformer Block。
