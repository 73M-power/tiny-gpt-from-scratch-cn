# 第 8 课：训练循环、验证集与过拟合

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. 训练集、验证集和测试集分别解决什么问题。
2. 一个训练 step 与一个 epoch 有什么区别。
3. `model.train()`、`model.eval()` 和 `torch.no_grad()` 各自控制什么。
4. 为什么评估时要平均多个随机 batch 的 loss。
5. 怎样根据 train loss 与 val loss 判断欠拟合、正常学习和过拟合。
6. 为什么训练 loss 持续下降并不保证模型越来越好。
7. 数据泄漏为什么会让验证结果失去意义。

## 1. 数据为什么要拆分

训练的真正目标不是背出已经见过的语料，而是学到可以迁移到未见文本的模式。因此至少需要两个互不混用的数据集合：

```text
训练集 train：用于计算梯度并更新参数
验证集 val：不更新参数，只用于估计模型在未参与训练的数据上的表现
```

更严格的项目通常还会有：

```text
测试集 test：模型和超参数都确定后，只用于最后一次独立报告
```

如果根据验证集反复选择模型、修改结构和调整超参数，验证集也会逐渐参与开发决策。因此最终报告通常还需要从未用于调参的测试集。

本教学项目只有 train/val，没有独立 test，这是为了保持代码最小化。

因此项目中的 `val` 应称为“验证集”，不是“测试集”。两者都不应参与常规参数更新，但验证集用于训练期间的模型选择与调参，测试集应留到开发结束后的最终独立评估。

## 2. 本项目怎样切分数据

代码：

```python
encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
split_at = int(0.9 * len(encoded))
splits = {
    "train": encoded[:split_at],
    "val": encoded[split_at:],
}
```

当前语料有 2319 个字符/token ID：

```text
train：前 2087 个，约 90%
val：  后  232 个，约 10%
```

它采用连续切分，而不是先生成所有窗口再随机拆分。`make_batch` 分别在各自 split 内部采样，因此不会生成跨越 train/val 分界线的窗口。

连续切分对语言序列有一个直观优点：验证集位于原文后部，更接近“用前面的文本训练，再检查后面的未见文本”。但如果前后文本主题或难度差别很大，验证结果也会受到分布差异影响。

### 教学简化说明

当前 tokenizer 由完整文本建立词表：

```python
tokenizer = CharTokenizer.from_text(text)
```

所以验证集出现过哪些字符参与了词表建立，但验证字符的排列和 next-token 答案没有参与参数更新。这是教学项目的简化。更严格的流程会只用训练集拟合 tokenizer，并把验证集中未见字符映射为 unknown token。

## 3. make_batch 在两个 split 中独立工作

```python
starts = torch.randint(number_of_start_positions, (batch_size,))
x = torch.stack([data[start : start + block_size] for start in starts])
y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
```

传入 `splits["train"]` 时，只能采样训练区域；传入 `splits["val"]` 时，只能采样验证区域。

当前默认：

```text
B = 16
T = 64
```

每个训练 step 同时产生：

```text
B×T = 16×64 = 1024 个 next-token prediction
```

训练区域可能的窗口起点数：

```text
2087 - 64 = 2023
```

验证区域可能的窗口起点数：

```text
232 - 64 = 168
```

随机采样允许重复，且相邻窗口高度重叠。因此“800 steps 看了 819200 个预测位置”不表示看到了 819200 个互不重复的数据：

```text
800×16×64 = 819200
```

在如此小的语料上反复采样，模型很容易逐渐记忆训练文本。

## 4. Step 与 Epoch

一个 step 表示：

```text
抽一个 batch
→ forward
→ backward
→ optimizer.step()
```

一个 epoch 通常表示完整遍历训练集一次。

但本项目不是按固定顺序、不重复地遍历所有样本，而是从可能窗口中随机有放回采样。因此 step 是最准确的训练进度单位，没有严格的“这一轮所有样本恰好看过一次”。

可以粗略估算：

```text
2023 个可能起点 ÷ batch size 16 ≈ 126 steps
```

约相当于采样一轮窗口数量，但因为会重复或漏掉窗口，不能当作严格 epoch。

## 5. 训练循环做什么

项目核心结构：

```python
model.train()
for step in range(args.steps + 1):
    if step % args.eval_interval == 0 or step == args.steps:
        losses = estimate_loss(...)

    if step == args.steps:
        break

    inputs, targets = make_batch(splits["train"], ...)
    _, loss = model(inputs, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

只有这一行采样训练数据：

```python
make_batch(splits["train"], ...)
```

只有后面的 `optimizer.step()` 修改模型参数。验证 loss 不参与 backward，也不会直接改变模型。

`range(args.steps + 1)` 多包含一个编号，是为了在最终 step 评估并保存，而不是多更新一次。到：

```python
if step == args.steps:
    break
```

时会退出，因此 `--steps 800` 恰好执行 800 次参数更新。

## 6. estimate_loss 为什么要平均

代码：

```python
@torch.no_grad()
def estimate_loss(...):
    model.eval()
    result = {}
    for split_name, split_data in splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = make_batch(split_data, ...)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[split_name] = sum(losses) / len(losses)
    model.train()
    return result
```

单个随机 batch 可能刚好简单或困难，loss 噪声较大。因此 train 与 val 都随机抽 `eval_iters` 个 batch 后取平均。

```text
eval_iters 小：评估快，但曲线抖动大
eval_iters 大：评估更稳定，但花费更多计算
```

`eval_iters` 不是另一种 learning rate：

```text
learning_rate：影响 optimizer.step() 修改参数的步幅
eval_iters：只影响一次评估抽取多少个 batch 来估计平均 loss
```

把 `eval_iters` 从 1 改为 20 不会让每次参数更新变慢或变小，也不会直接改变训练方向；它只让每次评估计算更多样本，因此评估阶段更耗时、结果通常更稳定。

项目默认：

```text
eval_interval = 100
eval_iters    = 20
```

也就是每 100 个更新 step 做一次评估，每个 split 平均 20 个随机 batch。

打印的 `train loss` 不是刚才用于更新的那一个训练 batch loss，而是在训练 split 上额外抽样、使用评估模式计算出的平均 loss。这样它与 val loss 的测量方式更可比。

## 7. train()、eval() 与 no_grad() 不相同

### model.train()

设置训练模式。本项目中主要影响 Dropout：

```text
Dropout 开启，随机丢弃部分特征
```

本项目默认 `dropout=0.1`。训练时，每次 forward 都会重新随机选择符合条件的中间激活元素，把其中约 10% 暂时置零，并对保留元素做相应缩放以维持期望尺度。它不会永久删除模型权重，也不是固定删除 10% 的 token。

本项目在多个位置使用 Dropout：

```text
token + position embedding 后的特征
softmax 后的 Attention weights
Attention 输出投影后的特征
FFN 输出后的特征
```

每次训练 forward 使用不同的随机 mask，迫使模型不能长期依赖某一条特征或连接路径，从而起到正则化、缓解过拟合的作用。

### model.eval()

设置评估模式：

```text
Dropout 关闭
```

评估与生成时不再随机丢弃这些激活，Dropout 相当于恒等映射；训练阶段已经对保留元素做过尺度补偿。

`model.eval()` 不会自动关闭梯度，也不会阻止参数更新；它只切换某些模块的行为。

### torch.no_grad()

关闭 autograd 计算图记录：

```text
减少内存和额外计算
不能对其中得到的 loss 做正常 backward
```

`torch.no_grad()` 不会自动关闭 Dropout。

因此验证通常同时需要：

```text
model.eval() + torch.no_grad()
```

评估结束后项目调用：

```python
model.train()
```

恢复训练模式，确保下一轮参数更新时 Dropout 再次生效。

## 8. Train loss 与 Val loss 各回答什么

```text
train loss：模型对参与训练的数据规律拟合得怎样
val loss：模型把规律迁移到未参与参数更新的数据上怎样
```

两者之间的差距常称为 generalization gap：

```text
gap = val loss - train loss
```

gap 不是越接近 0 就绝对越好，但持续扩大通常说明模型越来越专门适配训练数据。

## 9. 如何判断曲线

| 现象 | 常见解释 |
| --- | --- |
| train 和 val 都很高且继续下降 | 模型仍在学习，当前可能欠拟合 |
| train 和 val 一起下降，差距较稳定 | 模型正在学习可泛化规律 |
| train 持续下降，val 停止下降并回升 | 典型过拟合信号 |
| train 和 val 都很高且长期不变 | 模型能力、数据、学习率或优化过程可能有问题 |
| val 偶尔比 train 低 | 可能是随机噪声、验证数据更容易或测量模式不同，不必立刻判为错误 |

不能仅凭相邻两个评估点下结论。验证 loss 是抽样估计值，需要结合多个点、评估方差和实际任务表现判断。

## 10. 本项目的 800-step 实验

命令：

```bash
../../.venv/bin/python train.py \
  --steps 800 \
  --eval-interval 100 \
  --eval-iters 20 \
  --device cpu \
  --output /tmp/tinygpt-lesson8.pt
```

模型：

```text
参数量：135040
B=16，T=64，词表 V=490
```

实际结果：

| step | train loss | val loss | val-train gap |
| ---: | ---: | ---: | ---: |
| 0 | 6.2004 | 6.1989 | -0.0015 |
| 100 | 4.9085 | 5.2832 | 0.3747 |
| 200 | 3.9763 | 4.7016 | 0.7253 |
| 300 | 3.2948 | 4.2544 | 0.9596 |
| 400 | 2.7571 | 4.0996 | 1.3425 |
| 500 | 2.3332 | 3.9795 | 1.6463 |
| 600 | 1.9843 | 3.8979 | 1.9136 |
| 700 | 1.7054 | 4.0039 | 2.2985 |
| 800 | 1.4974 | 4.0217 | 2.5243 |

可以分为两个阶段：

```text
step 0→600：train 与 val 总体都下降，模型学到了可迁移规律
step 600→800：train 继续下降，但 val 回升，泛化差距继续扩大
```

这一次实验中，已观察评估点的最低 val loss 出现在 step 600 附近。由于评估有随机抽样噪声，不能把 600 当成数学上精确的唯一最优点，但它明显优于只根据最低 train loss 选择 step 800。

当前代码只在训练结束时保存最终模型，没有自动保存最低 val loss 对应的 checkpoint。因此如果目标是获得最佳泛化模型，通常还会增加“保存 best validation checkpoint”和 early stopping。

Checkpoint 是训练状态在某个 step 的磁盘快照，不只是一个时间标签。最基本的 checkpoint 会保存当时的模型参数；还可以同时保存 optimizer 状态、训练 step、配置和 tokenizer。所谓“step 600 的 checkpoint”，就是完成约 600 次参数更新时把这些状态写入文件，之后可以加载该快照用于推理或继续训练。

## 11. 什么是过拟合

过拟合表示模型在训练数据上表现越来越好，但这种改进没有迁移到未见数据，甚至损害未见数据表现。

本实验中：

```text
模型参数约 13.5 万
训练语料只有 2087 个字符
窗口反复重叠采样
```

模型容量相对数据量很大，反复训练后容易记住训练语料中的具体片段。

过拟合不等于训练代码出错。它是模型容量、训练时长、数据规模和正则化共同作用的结果。

## 12. 怎样缓解过拟合

常见方法：

1. 增加数量更多、分布更丰富、质量更高的训练数据。
2. 在 val loss 不再改善时提前停止训练，即 early stopping。
3. 保存最低 val loss 对应的 checkpoint，而不是盲目保存最后一步。
4. 增大合理的 Dropout 或 weight decay 等正则化强度。
5. 减小层数、隐藏维度等模型容量。
6. 清理重复、泄漏和低质量数据。
7. 使用更可靠、更有代表性的验证集。

每次实验应只改变一个主要变量，否则很难判断改善来自哪里。

不能通过“继续在验证集上训练”解决过拟合，因为这样验证集就变成了训练集，失去了独立评估价值。

## 13. 数据泄漏

下面行为会造成或接近数据泄漏：

- 使用验证 batch 执行 `backward()` 和 `optimizer.step()`。
- 先生成高度重叠的窗口，再把相邻窗口随机分到 train 与 val。
- 根据测试集结果反复调参。
- 训练语料中包含验证问题及其答案的重复副本。
- 在时间序列任务中用未来数据训练，再评估过去或当前。

发生泄漏后，val loss 看起来可能很好，但它不再可靠地代表真正未见数据。

## 14. 随机种子与可复现性

项目设置：

```python
random.seed(args.seed)
torch.manual_seed(args.seed)
```

同一环境中有助于复现初始化、batch 采样和 Dropout 随机序列。但不同硬件、PyTorch 版本或某些并行算子仍可能产生细微差异。

可复现实验还应记录：

- 代码提交哈希。
- 数据版本和切分方式。
- 模型配置。
- 优化器和学习率。
- batch size、block size、训练 steps。
- 随机种子与设备。

## 15. 完整训练状态流

```text
训练 split
→ make_batch
→ model.train()
→ forward loss
→ backward
→ AdamW step
→ 参数更新

每隔 eval_interval：
train split ─┐
             ├→ model.eval() + no_grad() → 平均 loss → 只记录
val split ───┘
→ model.train()
→ 继续训练
```

验证的作用是观察和选择，不是参与本轮参数更新。

## 常见误区

1. train loss 最低不代表模型泛化最好。
2. val loss 不参与 backward；如果参与，它就不再是独立验证。
3. `model.eval()` 与 `torch.no_grad()` 作用不同，验证通常两者都需要。
4. 一个 step 是一次参数更新，不等于完整看过一次训练集。
5. 随机窗口训练中没有严格 epoch，窗口会重复且相互重叠。
6. 单个验证 batch 的 loss 噪声很大，应该平均多个 batch。
7. 训练 loss 下降、验证 loss 上升是过拟合信号，不是“验证数据也应该加入训练”的理由。
8. 最后一轮 checkpoint 不一定是验证表现最好的 checkpoint。
9. val 是验证集，不等同于最终测试集；验证集用于训练期间选择模型，测试集用于最终独立报告。
10. Dropout 临时置零的是当前 forward 的部分中间激活或注意力连接，不会永久删除参数。
11. `eval_iters` 控制评估样本数，不是 learning rate，也不参与 `optimizer.step()`。

## 验收题

1. train split 和 val split 中，哪一个可以参与 `optimizer.step()`？为什么？
2. 如果 train loss 持续下降，而 val loss 先下降后连续回升，这是什么信号？应该优先选择最后一步还是 val loss 较低的 checkpoint？
3. `model.eval()` 与 `torch.no_grad()` 分别做什么？为什么验证时通常两者都要用？
4. `eval_iters` 从 1 增大到 20，评估结果和计算成本通常怎样变化？
5. 默认 `B=16、T=64`，一个训练 step 计算多少个 next-token prediction？800 steps 的计算位置数是多少？这些位置是否都互不重复？
6. 为什么不能为了降低 val loss，直接对 val batch 执行 backward 和 step？

## 验收结论

学员已经能够区分三个容易混淆的对象：

```text
eval_iters：增加参与平均的评估 batch 数，通常降低随机抽样方差；不改变参数更新
Dropout：训练 forward 时暂时随机置零部分中间激活，降低特征过度共适应；不永久删除参数
checkpoint：某个训练 step 对应的真实状态快照文件，不只是 step 数字或时间标签
```

“平均更多 batch”只能让当前验证分布下的 loss 估计通常更稳定，不保证数据切分本身没有偏差。Dropout 也不是消除模型内部的所有依赖关系，而是降低模型对少数特征或连接路径的过度依赖；评估和推理模式下使用完整激活。
