# 第 6 课：Logits、Softmax 与交叉熵

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. `lm_head` 怎样把 `(B,T,C)` 变成 `(B,T,V)`。
2. logits 为什么是原始分数而不是概率。
3. softmax 为什么沿最后的词表维 V 计算。
4. 单个位置的交叉熵为什么等于正确 token 概率的负对数。
5. `F.cross_entropy` 为什么接收原始 logits，而不是手动 softmax 后的概率。
6. 为什么训练时要把 `(B,T,V)` 和 `(B,T)` 分别 reshape 成 `(B×T,V)` 与 `(B×T)`。

## 1. 从隐藏状态到词表分数

所有 Transformer Block 结束后，项目执行：

```python
logits = self.lm_head(self.final_norm(x))
```

shape 变化：

```text
x                           (B,T,C)
final_norm(x)               (B,T,C)
lm_head                     C → V
logits                      (B,T,V)
```

本项目中 `V=490`，所以每个 token 位置都得到 490 个分数：每个分数对应词表中的一个候选 token。

`lm_head` 的定义：

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

PyTorch 的 Linear 权重 shape 为 `(输出维度, 输入维度)`，因此这里是 `(V,C)`。对于位置 `(b,t)` 和候选 token `v`：

```text
logits[b,t,v] = dot(hidden[b,t,:], lm_head.weight[v,:])
```

也就是说，`lm_head` 用同一个隐藏状态分别给词表中的 V 个候选 token 打分。

本项目还使用了权重绑定：

```python
self.lm_head.weight = self.token_embedding.weight
```

输入时，这个矩阵把 token ID 映射为 C 维向量；输出时，同一个矩阵又作为 V 个候选 token 的评分向量。这样能减少参数量，并让输入表示与输出评分共享同一套 token 特征空间。

## 2. Logits 不是概率

假设词表只有三个 token：

```text
token 0 = 甲
token 1 = 乙
token 2 = 丙
```

某个位置的输出为：

```text
logits = [2.0, 1.0, 0.0]
```

它表达的是模型当前更偏向：

```text
甲 > 乙 > 丙
```

但 logits 还不是概率，因为：

- 可以是负数。
- 可以大于 1。
- 所有值不需要相加等于 1。
- 它们只表示候选项之间的相对评分。

给所有 logits 同时加上相同常数不会改变 softmax 概率：

```text
[2,1,0] 与 [102,101,100] 得到相同概率
```

因此绝对数值不是重点，候选 token 之间的相对差距才是重点。

## 3. Softmax：把 V 个分数变成概率

softmax 沿词表维 V 计算：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

实际实现通常先减去最大 logit，提高数值稳定性：

```text
p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

两种公式数学结果相同。

对：

```text
logits = [2.0, 1.0, 0.0]
```

有：

```text
exp(2) ≈ 7.389
exp(1) ≈ 2.718
exp(0) = 1
总和   ≈ 11.107
```

所以：

```text
P(甲) ≈ 7.389 / 11.107 = 0.6652
P(乙) ≈ 2.718 / 11.107 = 0.2447
P(丙) ≈ 1.000 / 11.107 = 0.0900
```

这里必须区分“数组索引”和“数组中的数值”：

| 词表索引/target ID | token | 该索引处的 logit | softmax 后该索引处的概率 |
| ---: | --- | ---: | ---: |
| 0 | 甲 | 2.0 | 0.6652 |
| 1 | 乙 | 1.0 | 0.2447 |
| 2 | 丙 | 0.0 | 0.0900 |

`target=0` 表示选择数组索引 0，所以读取第一个概率 `0.6652`；它不表示寻找“数值等于 0 的 logit”。数值为 `0.0` 的 logit 恰好位于索引 2，因此对应的是 `target=2` 和概率 `0.0900`。

可以直接记成：

```text
target=y → 使用 probabilities[y]
```

softmax 输出满足：

```text
每个概率在 0 到 1 之间
所有候选 token 的概率之和为 1
```

对于 logits shape `(B,T,V)`：

```python
probabilities = F.softmax(logits, dim=-1)
```

`dim=-1` 表示沿最后的 V 维处理：每个 batch、每个位置独立得到一份包含 V 个候选 token 的概率分布。不能沿 T 维 softmax，因为 T 表示不同位置，而不是同一位置的候选答案。

## 4. 交叉熵：正确答案获得了多少概率

对于单个位置，真实的下一个 token ID 记作 `y`。交叉熵为：

```text
loss = -log(P(y))
```

仍使用：

```text
P(甲)=0.6652
P(乙)=0.2447
P(丙)=0.0900
```

如果真实答案是甲，即 `target=0`：

```text
loss = -log(0.6652) ≈ 0.4076
```

如果真实答案是乙，即 `target=1`：

```text
loss = -log(0.2447) ≈ 1.4076
```

如果真实答案是丙，即 `target=2`：

```text
loss = -log(0.0900) ≈ 2.4076
```

比较 loss 时必须比较“各自正确 token 的概率”：

```text
target=0：正确概率 0.6652 → loss 0.4076
target=1：正确概率 0.2447 → loss 1.4076
target=2：正确概率 0.0900 → loss 2.4076
```

`target=1` 的 loss 比 `target=0` 大，是因为 `0.2447 < 0.6652`。`0.2447 > 0.0900` 只能说明它的 loss 比 `target=2` 小。这里的 `0.2447` 应称为 softmax 概率，不是持久化的模型权重。

因此：

```text
正确 token 的概率越接近 1 → loss 越接近 0
正确 token 的概率越接近 0 → loss 越大
```

虽然公式最后读取的是正确类别的概率，但这个概率的分母包含所有 V 个 logits，因此错误候选的分数也会影响 loss。

## 5. PyTorch 为什么直接接收 logits

项目代码：

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

这里没有先写：

```python
probabilities = F.softmax(logits, dim=-1)
```

原因是 `F.cross_entropy` 已经在内部组合了：

```text
log_softmax + negative log likelihood
```

它要求：

- 第一个参数是未经 softmax 的原始 logits。
- 第二个参数是正确类别的整数 ID，而不是 one-hot，也不是概率。

直接组合计算比“先 softmax，再取 log”更稳定。训练时不要把手动 softmax 后的概率再次传给 `F.cross_entropy`。

## 6. 为什么要 reshape

模型输出：

```text
logits  (B,T,V)
targets (B,T)
```

PyTorch 分类交叉熵可以把每个 token 位置看成一个分类样本，因此项目把 B 和 T 合并：

```text
logits.reshape(-1, V) → (B×T,V)
targets.reshape(-1)   → (B×T)
```

例如：

```text
B=2, T=3, V=5

logits:  (2,3,5) → (6,5)
targets: (2,3)   → (6)
```

这表示一共有 6 个位置需要预测，每个位置都要在 5 个 token 中选择正确答案。reshape 只改变观察数据的 shape，不会打乱 logits 与 target 的对应顺序。

`F.cross_entropy` 默认对这 `B×T` 个位置的 loss 取平均，最终得到一个标量：

```text
loss shape = ()
```

## 7. 为什么一次可以训练 T 个预测

如果输入和目标为：

```text
input : 我 爱 学 习
target: 爱 学 习 。
```

模型一次 forward 会同时产生：

```text
看到“我”       → 预测“爱”
看到“我爱”     → 预测“学”
看到“我爱学”   → 预测“习”
看到“我爱学习” → 预测“。”
```

因果 mask 保证前面位置看不到未来 token，所以训练时可以并行计算 T 个合法的 next-token prediction。

训练和生成使用 logits 的方式不同：

```text
训练：使用所有 B×T 个位置的 logits 计算平均 loss
生成：通常只取最后一个位置 logits[:, -1, :] 预测下一个 token
```

## 8. 随机模型的 loss 基线

如果模型对 V 个 token 完全没有偏好，每个 token 的概率都是 `1/V`：

```text
loss = -log(1/V) = log(V)
```

本项目 `V=490`：

```text
log(490) ≈ 6.1944
```

因此随机初始化模型的 loss 在 6.2 左右是合理现象。训练有效时，平均 loss 应整体下降；但单个 step 上下波动并不代表训练一定失败。

## 9. Loss 还不会自己更新参数

交叉熵只负责把一整批预测的好坏汇总成标量 loss：

```text
logits + targets → loss
```

模型参数真正改变还需要后续步骤：

```text
loss.backward()  → 计算参数梯度
optimizer.step() → 根据梯度更新参数
```

这是下一课“反向传播与 AdamW”的重点。

## 完整 Shape

```text
Block 输出 x                         (B,T,C)
final_norm(x)                       (B,T,C)
lm_head                             (B,T,V)
logits.reshape(-1,V)                (B×T,V)
targets                             (B,T)
targets.reshape(-1)                 (B×T)
F.cross_entropy                     标量 loss，shape ()
```

## 常见误区

1. logits 不是概率，也不是 Attention weights。
2. softmax 不会让模型自动选出正确答案，只是把相对分数转成概率。
3. 训练时 `F.cross_entropy` 接收原始 logits，不要提前手动 softmax。
4. `target` 是正确 token 的整数 ID，不是模型预测出的 ID。
5. loss 是衡量整批预测好坏的标量，不是模型参数。
6. loss 较低表示模型给真实答案分配了更高概率，不等于每个位置都预测正确。
7. 比较两个 target 的 loss 时，要比较两个 target 各自得到的概率；负对数函数会让更小的正确概率对应更大的 loss。
8. `target=0` 中的 0 是词表索引，不是要求 logit 的数值等于 0；loss 读取的是 `probabilities[target]`。

单个分类例子中的 `target` 是一个整数索引，例如 `target=2`。完整 batch 中的 `targets` 才是 shape 为 `(B,T)` 的张量，其中每个位置保存一个正确 token ID。

## 验收题

1. 若 `logits.shape=(2,4,490)`，三个维度分别代表什么？softmax 应沿哪个维度计算？
2. 对 `logits=[2,1,0]`，softmax 约为 `[0.6652,0.2447,0.0900]`。如果 `target=1`，loss 约是多少？
3. 为什么不应该先手动 softmax，再把结果传给 `F.cross_entropy`？
4. `logits=(3,10,490)`、`targets=(3,10)`，送入交叉熵前分别 reshape 成什么 shape？
5. `loss.backward()` 与 `optimizer.step()` 分别负责什么？

## 验收结论

对于：

```text
索引          0       1       2
logits      [2.0,    1.0,    0.0]
probability [0.6652, 0.2447, 0.0900]
```

`target=2` 表示读取数组索引 2，而不是寻找数值等于 2 的元素。因此正确概率为 `probabilities[2]=0.0900`，单位置 loss 为 `-log(0.09)≈2.4076`。学员已经能够区分 target ID、logit 数值与 softmax 概率，并能跟踪交叉熵前后的 shape。
