# 第 7 课：梯度、反向传播与 AdamW

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. 梯度的正负和大小分别表示什么。
2. `loss.backward()` 为什么只计算梯度，不更新参数。
3. `optimizer.step()` 怎样使用梯度更新持久模型参数。
4. 为什么每个训练 step 都要清理旧梯度。
5. 梯度裁剪解决什么问题。
6. AdamW 与最基础梯度下降有什么关系。
7. forward、backward、参数更新中的临时量和持久量分别是什么。

## 1. 一次训练 step 的完整主线

项目训练循环最关键的代码：

```python
inputs, targets = make_batch(...)
_, loss = model(inputs, targets)

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

分别负责：

```text
make_batch       取得训练输入和正确答案
model(...)       forward：计算隐藏状态、logits 和 loss
zero_grad        清理上一步残留的梯度
backward         从 loss 反向计算每个参数的梯度
clip_grad_norm_  梯度过大时按比例缩小
step             AdamW 根据梯度真正修改模型参数
```

一句话区分：

```text
forward：这次预测得怎么样？
backward：每个参数该往哪个方向改？
step：真正把参数改掉。
```

## 2. 梯度是什么

假设模型只有一个参数 `w`，loss 写作：

```text
L(w)
```

梯度：

```text
dL/dw
```

表示在当前 `w` 附近，把 `w` 增大一点时，loss 会怎样变化：

- 梯度为正：增大 `w` 会让 loss 增大，所以梯度下降倾向于减小 `w`。
- 梯度为负：增大 `w` 会让 loss 减小，所以梯度下降倾向于增大 `w`。
- 梯度绝对值大：loss 对这个参数在当前点更敏感。
- 梯度接近 0：在当前点小幅改变这个参数，对 loss 的一阶影响较小。

梯度不是“正确参数值”，也不是应该直接填入参数的修正答案。它是当前位置的局部斜率。

负梯度也不能简单解释成“数据偏小”或“参数一定偏小”。它只说明在当前参数点、当前 loss 曲面上，沿这个参数的正方向移动一小步会使 loss 下降。例如 `L=(w-10)²` 在 `w=9` 时梯度为负，适合增大 w；但这个判断来自 loss 的局部斜率，不来自数字 9 本身算大还是小。

## 3. 手算一个参数的训练

建立最小模型：

```text
x = 2
target = 6
prediction = w × x
loss = (prediction - target)²
```

初始参数：

```text
w = 1
```

### 3.1 Forward

```text
prediction = 1 × 2 = 2
loss = (2 - 6)² = 16
```

### 3.2 Backward

使用链式法则：

```text
dL/dw
= dL/dprediction × dprediction/dw
= 2(prediction-target) × x
= 2(2-6) × 2
= -16
```

梯度是 `-16`。它的含义不是“把 w 设置为 -16”，而是：当前点如果增大 `w`，loss 会下降。

### 3.3 Gradient descent step

最基础的梯度下降：

```text
w_new = w_old - learning_rate × gradient
```

设学习率 `lr=0.1`：

```text
w_new = 1 - 0.1×(-16)
      = 2.6
```

重新 forward：

```text
prediction = 2.6×2 = 5.2
loss = (5.2-6)² = 0.64
```

这个例子中，loss 从 16 降到了 0.64。

## 4. 学习率控制步幅

更新公式：

```text
parameter_new = parameter_old - lr × gradient
```

`lr` 即 learning rate：

- 太小：每次更新很小，训练可能非常慢。
- 太大：可能跨过较好的区域，使 loss 剧烈震荡甚至发散。
- 合适：沿整体下降方向逐步优化。

本项目默认：

```python
--learning-rate 3e-4
```

即：

```text
3e-4 = 0.0003
```

复杂神经网络不是简单碗形函数，单个 step 也不保证每次 loss 都下降；我们通常观察一段训练区间的整体趋势和验证集表现。

## 5. 链式法则与反向传播

Tiny GPT 的 forward 是一条很长的计算链：

```text
参数
→ embedding
→ Transformer Blocks
→ final_norm
→ lm_head
→ logits
→ cross_entropy
→ loss
```

反向传播沿相反方向应用链式法则：

```text
loss
→ cross_entropy 的梯度
→ logits 的梯度
→ lm_head 参数的梯度
→ 各 Transformer Block 参数的梯度
→ embedding 参数的梯度
```

如果：

```text
w → a → b → loss
```

那么：

```text
dLoss/dw
= dLoss/db × db/da × da/dw
```

反向传播不是“从答案重新运行一次模型”，而是复用 forward 时记录的运算关系和必要中间值，高效计算大量参数的导数。

## 6. PyTorch Autograd 做什么

PyTorch 中模型参数通常满足：

```python
parameter.requires_grad == True
```

forward 时，PyTorch 动态记录由这些张量参与的运算，形成计算图。执行：

```python
loss.backward()
```

后，每个参与 loss 计算的参数会在：

```python
parameter.grad
```

中得到梯度。

如果某参数 shape 为：

```text
parameter.shape = (490,64)
```

它的梯度通常具有相同 shape：

```text
parameter.grad.shape = (490,64)
```

每个参数元素都有自己的局部导数。

`loss.backward()` 的结果不是一个新的模型，也不会直接改动 `parameter.data`。它主要把计算结果累积到各参数的 `.grad` 中。

## 7. 为什么必须 zero_grad

PyTorch 默认累加梯度，而不是自动覆盖：

```text
已有 parameter.grad + 本次 backward 得到的梯度
```

在手算例子中，第一次梯度是：

```text
-16
```

更新到 `w=2.6` 后，新梯度是：

```text
-3.2
```

如果不清理旧梯度，第二次 backward 后会得到：

```text
-16 + (-3.2) = -19.2
```

这不是当前 batch 单独对应的梯度。

因此项目每一步执行：

```python
optimizer.zero_grad(set_to_none=True)
```

这个 optimizer 管理 `model.parameters()`，所以它会清理整个模型中由该 optimizer 管理的参数梯度，不是只清理当前 Block。要清除的是上一轮训练 batch 留下的 `.grad`，避免它与本轮 batch 新计算的梯度意外相加；隐藏状态和模型参数不会被这行代码清零。

`set_to_none=True` 会把旧 `.grad` 设置为 `None`，下一次 backward 再创建新梯度，通常更节省内存和写入操作。

项目把 `zero_grad` 写在 forward 之后、backward 之前，这是合法的：forward 不会读取上一轮参数梯度。也常见下面的顺序：

```text
zero_grad → forward → backward → step
```

关键要求是：在本轮 backward 前清掉不需要累积的旧梯度。

有些大模型训练会故意跨多个小 batch 累积梯度，以模拟更大的 batch；那是有意跳过部分 `zero_grad`，并不是忘记清理。

## 8. 梯度裁剪

项目在 backward 和 step 之间执行：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它先计算所有参数梯度的整体范数。如果整体范数不超过 1，就保持不变；如果超过 1，就按相同比例缩小这些梯度，使整体范数约为 1。

梯度裁剪的目的：防止偶发的巨大梯度造成一次过强的参数更新，使训练数值不稳定。

它不是：

- 把每个梯度元素单独截成 `[-1,1]`。
- 保证 loss 一定下降。
- 替代合适的学习率。

## 9. optimizer.step() 才真正更新参数

```python
optimizer.step()
```

会读取：

- 当前模型参数。
- `parameter.grad` 中的梯度。
- AdamW 为每个参数维护的优化器状态。
- learning rate、betas、epsilon、weight decay 等超参数。

然后原地修改持久模型参数。下一次 forward 使用的就是更新后的参数。

区分四个动作：

| 动作 | 计算隐藏状态 | 计算 `.grad` | 修改模型参数 | 清理 `.grad` |
| --- | --- | --- | --- | --- |
| `model(inputs, targets)` | 是 | 否 | 否 | 否 |
| `loss.backward()` | 否 | 是 | 否 | 否 |
| `optimizer.step()` | 否 | 否 | 是 | 否 |
| `optimizer.zero_grad()` | 否 | 否 | 否 | 是 |

## 10. AdamW 的直觉

最基础的 SGD 直接使用当前梯度：

```text
parameter ← parameter - lr × gradient
```

AdamW 会为每个参数维护两个移动平均：

```text
m：梯度的一阶移动平均，近似近期的主要方向
v：梯度平方的二阶移动平均，近似近期的变化尺度
```

简化公式：

```text
m_t = β₁m_(t-1) + (1-β₁)g_t
v_t = β₂v_(t-1) + (1-β₂)g_t²

parameter ← parameter
            - lr × m_hat / (sqrt(v_hat)+eps)
            - lr × weight_decay × parameter
```

其中 `m_hat`、`v_hat` 还包含训练初期的偏差修正。

直觉上：

- `m` 平滑单个 batch 的噪声，保留近期较稳定的更新方向。
- `v` 让不同参数根据自己的历史梯度尺度调整有效步幅。
- `eps` 防止分母过小或除零。
- weight decay 让参数持续受到独立的收缩约束，这也是 AdamW 中字母 W 的关键。

项目创建优化器：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
)
```

由于这里只显式传入学习率，本机 PyTorch 2.8 使用默认：

```text
betas        = (0.9, 0.999)
eps          = 1e-8
weight_decay = 0.01
```

不需要先背公式。当前最重要的是理解：AdamW 仍然使用 backward 得到的梯度，只是比基础 SGD 多保存了历史统计，并自适应调整每个参数的更新。

## 11. 临时量与持久量

| 对象 | 示例 shape | 在训练中的生命周期 |
| --- | --- | --- |
| 隐藏状态 | `(B,T,C)` | 当前 forward/backward 临时使用，之后通常释放 |
| logits | `(B,T,V)` | 当前 batch 临时使用 |
| loss | 标量 `()` | 当前 batch 临时使用 |
| Attention weights | `(B,N,T,T)` | 当前 forward 临时使用 |
| 参数 | `(V,C)`、`(C,C)` 等 | 跨 batch 持久存在，由 `step()` 更新 |
| 参数梯度 `.grad` | 与对应参数 shape 相同 | backward 产生，下一轮通常由 zero_grad 清理 |
| AdamW 状态 m/v | 与对应参数 shape 相同 | 跨训练 step 持久存在，由优化器维护 |

当前项目 checkpoint 保存了模型参数，但没有保存 AdamW 的 `optimizer.state_dict()`；因此它能加载模型进行推理，但不能带着完全相同的 AdamW 历史状态无缝继续训练。第 10 课会专门讨论 checkpoint。

## 12. 为什么验证时不需要 backward

项目的验证函数使用：

```python
@torch.no_grad()
def estimate_loss(...):
    ...
```

验证阶段只需要计算预测和 loss，不需要更新参数。因此关闭梯度记录可以减少内存占用和额外计算。

```text
训练：forward → backward → step
验证：forward，只读取 loss
推理：forward，只使用 logits/概率
```

## 13. 一次更新的精确顺序

```text
1. 读取 inputs、targets
2. forward 得到 loss
3. 清理上一轮 gradients
4. backward 填充每个 parameter.grad
5. 必要时裁剪 gradients
6. AdamW step 更新 parameters 和优化器状态
7. 下一 batch 使用新 parameters 再次 forward
```

注意：Block 内的残差连接更新的是当前输入的隐藏状态；`optimizer.step()` 更新的才是跨 batch 持久存在的模型参数。这是两个不同层面的“更新”。

## 常见误区

1. 梯度不是正确参数值，而是 loss 对当前参数的局部导数。
2. `loss.backward()` 不会更新模型参数，只会计算并累积 `.grad`。
3. `optimizer.step()` 不会重新计算梯度，它使用已经存在的 `.grad`。
4. `zero_grad()` 清理的是梯度，不会把模型参数清零。
5. 梯度为负不表示结果错误；在梯度下降公式中减去负数会让参数增大。
6. 梯度绝对值大不代表这个参数更重要，只表示当前点 loss 对它的一阶敏感度较大。
7. AdamW 不是绕过反向传播，它同样依赖 backward 计算的梯度。
8. 一个 step 的 loss 上升不等于训练必然失败，要观察整体趋势和验证集。
9. 负梯度不等于“数据或参数偏小”，它只描述当前点沿该参数正方向移动时 loss 的局部变化。
10. `optimizer.zero_grad()` 作用于该 optimizer 管理的全部参数梯度，不是只清理某一个 Transformer Block。

## 验收题

1. 在手算例子中，`w=1、x=2、target=6`，为什么梯度 `dL/dw=-16` 会让梯度下降增大 `w`，而不是减小 `w`？
2. `loss.backward()` 与 `optimizer.step()` 分别修改或产生了什么？
3. `zero_grad()` 清除的是模型参数、隐藏状态，还是参数梯度？为什么通常每步都要执行？
4. 如果参数 shape 为 `(490,64)`，它的 `.grad` 通常是什么 shape？
5. 请按正确顺序排列：`optimizer.step()`、forward、`loss.backward()`、梯度裁剪、`zero_grad()`。
6. AdamW 相比最基础的 SGD，额外维护了哪两类历史统计？

## 验收结论

对于基础梯度下降：

```text
parameter_new = parameter_old - learning_rate × gradient
```

当 `w=4、lr=0.1` 时：

```text
gradient=+3 → w_new=4-0.1×3    = 3.7
gradient=-3 → w_new=4-0.1×(-3) = 4.3
```

这说明梯度的符号描述当前点的局部上升方向；梯度下降通过减去梯度向反方向移动。学员已经能够区分 `loss.backward()` 计算 `.grad` 与 `optimizer.step()` 更新参数，并明确 `zero_grad()` 清除的是 optimizer 管理的全部参数梯度，而不是单个 Block 或模型参数。
