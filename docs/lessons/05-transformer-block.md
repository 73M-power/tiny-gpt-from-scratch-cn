# 第 5 课：Transformer Block

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. Attention 与 FFN 各自负责什么。
2. LayerNorm 对哪个维度进行归一化。
3. 残差连接为什么要求子层输入输出 shape 相同。
4. Pre-LayerNorm 的执行顺序。
5. FFN 为什么先把 C 扩展为 4C，再压回 C。
6. 一个 Block 为什么始终保持 `(B,T,C)`。

## 1. Block 的整体结构

项目中的完整代码只有两行：

```python
x = x + self.attention(self.ln_1(x))
x = x + self.feed_forward(self.ln_2(x))
```

展开后：

```text
输入 x
  ├─ LayerNorm → Attention ─┐
  └─────────────────────────+→ x₁

x₁
  ├─ LayerNorm → FFN ───────┐
  └─────────────────────────+→ 输出 x₂
```

可以用一句话记忆：

```text
Attention 在 token 之间通信
FFN 在单个 token 内加工特征
Residual 保留原信息
LayerNorm 稳定送入子层的数值尺度
```

为了避免代码中重复覆盖变量 `x` 带来的混淆，可以把同一个 Block 精确展开：

```python
x0 = x
n1 = ln_1(x0)
a = attention(n1)
x1 = x0 + a

n2 = ln_2(x1)
h = linear_1(n2)
h = gelu(h)
f = linear_2(h)
f = dropout(f)
x2 = x1 + f
```

其中 `a` 是 Attention 对隐藏状态提出的更新量，`f` 是 FFN 提出的更新量，`x2` 是 Block 输出的隐藏状态。

## 2. LayerNorm 做什么

项目定义两个独立的 LayerNorm：

```python
self.ln_1 = nn.LayerNorm(C)
self.ln_2 = nn.LayerNorm(C)
```

输入 shape 为 `(B,T,C)`。LayerNorm 对每个 batch 中每个 token 的最后 C 维独立归一化：

```text
x[b,t,:] → 计算这个 token 的 C 个特征的均值与方差
```

它不混合不同 token，也不依赖 batch 中其他样本。

简化公式：

```text
normalized = (x - mean) / sqrt(variance + eps)
output     = gamma × normalized + beta
```

`gamma` 和 `beta` 是长度为 C 的可训练参数，使模型可以学习合适的缩放和平移。一个 `LayerNorm(C)` 有 `2C` 个参数。

例如一个 token 的三维向量为 `[1,2,3]`，均值为 2；归一化会让特征大致变成以 0 为中心、尺度较稳定的数值，再由 gamma 和 beta 调整。

LayerNorm 的目的不是为了得到一份统计报告，也不是强迫模型永远保持标准正态分布。它让 Attention 或 FFN 在不同 token、层和训练阶段收到尺度更可控的输入，减少激活值过大或过小造成的 softmax 饱和、梯度不稳定和优化困难。可学习的 gamma 与 beta 又允许模型在归一化之后恢复真正有用的尺度和偏移。

LayerNorm 也不是把两个 token 放在一起比较。对于：

```text
token A = x[b,0,:]
token B = x[b,1,:]
```

它分别计算 A 的 C 维均值/方差和 B 的 C 维均值/方差。一个 token 的大数值不会直接参与另一个 token 的归一化。归一化后的向量在应用 gamma、beta 之前具有接近 0 的均值和接近 1 的方差，但这不等于各维数值服从正态分布。

LayerNorm 也不是 Linear：

```text
Linear：y=xWᵀ+b，使用可训练矩阵混合特征
LayerNorm：使用当前 token 自己的均值和方差重新缩放，再应用 gamma/beta
```

LayerNorm 的输出依赖输入本身计算出的方差和平方根，因此整体不是一个固定的线性变换。

## 3. Pre-LayerNorm

本项目在进入子层之前归一化：

```python
attention(self.ln_1(x))
feed_forward(self.ln_2(x))
```

因此称为 Pre-LayerNorm：

```text
x + Sublayer(LayerNorm(x))
```

另一种 Post-LayerNorm 写法是：

```text
LayerNorm(x + Sublayer(x))
```

Pre-LN 保留了一条未经 LayerNorm 改写的残差主路径，并通常更有利于深层模型中的梯度传播和训练稳定性。本项目只实现 Pre-LN。

## 4. 残差连接

第一条残差：

```python
x = x + self.attention(self.ln_1(x))
```

可以写成：

```text
新 x = 原 x + Attention 提议的更新量
```

第二条残差：

```python
x = x + self.feed_forward(self.ln_2(x))
```

表示：

```text
新 x = 原 x + FFN 提议的更新量
```

残差连接的两个重要作用：

1. 信息可以沿恒等路径直接向后传递，子层只需要学习“应该修改多少”。
2. 反向传播时梯度拥有直接路径，更容易穿过许多层。

逐元素相加要求两边 shape 完全相同，因此 Attention 和 FFN 都必须从 `(B,T,C)` 回到 `(B,T,C)`。

残差连接发生在一次 forward 的层与层之间，它更新的是当前样本的隐藏状态，不是直接修改持久化的模型参数：

```text
层间更新隐藏状态：x_next = x + F(x)
训练更新模型参数：optimizer.step() 修改 W、bias 等参数
```

反向传播时：

```text
d(x + F(x))/dx = I + dF(x)/dx
```

恒等项 I 为梯度提供直接通道，即使 F 分支的梯度较弱，梯度仍能沿残差主路径传播。

隐藏状态并非“一过当前层立即消失”：它会作为下一层输入，并在训练时由 autograd 暂时保留反向传播需要的信息；本次 forward/backward 完成后才通常被释放。参数则跨 batch 持续存在，每次 `optimizer.step()` 后成为下一步训练的当前参数状态，并可写入 checkpoint。

## 5. FFN：逐 token 的特征加工

项目代码：

```python
self.net = nn.Sequential(
    nn.Linear(C, 4 * C),
    nn.GELU(),
    nn.Linear(4 * C, C),
    nn.Dropout(dropout),
)
```

shape：

```text
(B,T,C)
→ Linear   (B,T,4C)
→ GELU     (B,T,4C)
→ Linear   (B,T,C)
→ Dropout  (B,T,C)
```

当前 `C=64`：

```text
(B,T,64) → (B,T,256) → (B,T,64)
```

FFN 的线性层只处理最后一个维度，使用同一套参数独立处理每个 token。它不会在 T 个位置之间交换信息。

## 6. Attention 与 FFN 的分工

```text
Attention：沿 T 维汇总其他 token 的 Value，负责位置之间通信
FFN：沿 C 维重新组合和非线性变换特征，负责每个位置内部计算
```

只有 Attention，没有 FFN，模型虽然能读取上下文，却缺少足够的逐位置非线性特征变换能力。

只有 FFN，没有 Attention，每个位置只能独立加工自己，无法读取其他 token。

Transformer Block 把“通信”和“计算”组合在一起。

## 7. 为什么扩展到 4C

第一层把特征从 C 扩展到 4C，为非线性变换提供更大的中间空间；第二层再压回 C，以便继续残差连接和堆叠 Block。

`4C` 是经典 Transformer/GPT 中常见的设计比例，不是数学上唯一正确的选择。现代模型也可能使用其他比例和 SwiGLU 等不同 FFN 结构。

## 8. GELU 为什么重要

如果连续堆叠的线性层之间没有非线性激活：

```text
Linear₂(Linear₁(x))
```

整体仍可以合并为一个线性变换，表达能力有限。

GELU 引入非线性，使 FFN 能学习更复杂的特征组合。它的输出 shape 不变，也没有可训练参数。

GELU 不是在保持数学等价的前提下改变模型，而是专门打破两个 Linear 可以合并的等价性：

```text
无激活：Linear₂(Linear₁(x)) 仍等价于一个 Linear
有 GELU：Linear₂(GELU(Linear₁(x))) 通常无法合并成一个 Linear
```

两个 Linear 也各有作用。第一层学习把 C 个输入特征组合成 4C 个中间特征，第二层学习把这些经过非线性筛选的中间特征重新组合回 C 维：

```text
h = xW₁ᵀ + b₁
a = GELU(h)
y = aW₂ᵀ + b₂
```

每个 Linear 的输出维度都是输入各维度的可学习加权和，因此负责跨通道混合特征；GELU 根据当前数值进行非线性调节。二者承担不同职责。

## 9. 参数量

当前 `C=64`。

两个 LayerNorm：

```text
2 × (gamma 64 + beta 64) = 256
```

Attention 的四个无 bias 投影：

```text
4 × 64² = 16,384
```

FFN 使用默认 bias：

```text
第一层权重 64×256  = 16,384
第一层 bias         =    256
第二层权重 256×64  = 16,384
第二层 bias         =     64
FFN 合计            = 33,088
```

一个 Block 合计：

```text
16,384 + 33,088 + 256 = 49,728
```

Dropout 和 GELU 没有可训练参数。

## 10. 多层堆叠

项目使用：

```python
self.blocks = nn.ModuleList(
    [TransformerBlock(config) for _ in range(config.n_layer)]
)
```

当前 `n_layer=2`，所以一个 token 表示会依次通过两个不同参数的 Block。每层 shape 都保持 `(B,T,C)`，但内部数值不断吸收上下文并转换特征。

项目的 forward 使用了 Python 循环：

```python
for block in self.blocks:
    x = block(x)
```

因此可以把它理解为“按顺序遍历所有 Block”，但不能理解为传统意义上“同一个 Block 不断循环，直到得到答案”：

- `self.blocks` 中的每个 Block 都是不同的模块实例，拥有各自的参数。
- 一次 forward 中，隐藏状态依次通过第 1 层、第 2 层，直到第 `n_layer` 层，每层通常只执行一次。
- forward 期间只计算隐藏状态，不会在经过每个 Block 时直接更新模型参数。
- 训练循环会对不同 batch 重复执行完整 forward/backward，并由优化器更新参数。
- 生成循环会为了产生后续 token 重复使用整个模型；使用 KV Cache 时可以复用过去 token 的 K/V。

所以更准确的叫法是“有限深度的顺序堆叠”。代码用循环简洁地遍历这组层，不代表网络结构是共享参数的循环神经网络。

## 11. Block 输出还不是答案

Block 最终得到的 `x2` 是临时隐藏状态，不是模型权重，也不是最终 token 答案。完整模型还要继续：

```text
所有 Transformer Block 的输出 (B,T,C)
→ final LayerNorm             (B,T,C)
→ lm_head                     (B,T,V)
→ logits
```

训练时 logits 与 targets 计算 loss，再由反向传播和优化器修改持久模型参数；生成时取最后位置的 logits，经过 softmax/采样选择下一个 token。

需要区分四种常被简称为“权重”的对象：

| 对象 | Shape 示例 | 是否持久保存 | 作用 |
| --- | --- | --- | --- |
| 模型参数 W | `(C,C)` 等 | 是 | 训练学习并写入 checkpoint |
| Attention weights | `(B,N,T,T)` | 否 | 当前输入中决定读取各位置的比例 |
| 隐藏状态 x | `(B,T,C)` | 否 | 当前输入在各层之间传递的表示 |
| logits | `(B,T,V)` | 否 | 对词表中下一个 token 的原始预测分数 |

## 完整 Shape

```text
x                         (B,T,C)
ln₁(x)                    (B,T,C)
attention(ln₁(x))         (B,T,C)
x + attention             (B,T,C)
ln₂(x)                    (B,T,C)
FFN 第一层                 (B,T,4C)
GELU                      (B,T,4C)
FFN 第二层                 (B,T,C)
x + FFN                   (B,T,C)
```

## 验收题

1. Attention 和 FFN 分别沿哪个方向处理信息？
2. LayerNorm 对 `(B,T,C)` 的哪个维度计算均值和方差？
3. 为什么残差连接要求 Attention 和 FFN 最终回到 `(B,T,C)`？
4. `C=128` 时，FFN 中间层 shape 的最后一维是多少？
5. 为什么两个 Linear 中间必须加入 GELU 等非线性函数？
6. 本项目是 Pre-LN 还是 Post-LN？从代码哪里看出来？
7. 残差连接怎样帮助信息和梯度穿过深层网络？

## 验收结论

`x2 = x1 + FFN(LayerNorm(x1))` 中的 `x2` 是 shape 为 `(B,T,C)` 的隐藏状态：它由当前输入临时计算得到，将继续送入下一个 Block 或最终 LayerNorm。它不是 checkpoint 中持久保存的模型参数，不是 `(B,N,T,T)` 的 Attention weights，也还不是 `(B,T,V)` 的 logits。所有 Block 结束后，还要经过 `final_norm` 和 `lm_head` 才得到 logits。
