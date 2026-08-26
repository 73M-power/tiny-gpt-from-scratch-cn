# 第 9 课：自回归生成、Temperature 与 Top-k

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. 为什么训练可以并行预测 T 个位置，而生成必须逐 token 循环。
2. 为什么生成时只使用最后一个位置的 logits。
3. temperature 怎样改变概率分布，但不改变正温度下的 logit 排名。
4. top-k 怎样把候选范围限制为分数最高的 k 个 token。
5. `torch.multinomial` 与 `argmax` 的区别。
6. prompt 超过 context window 或持续生成后，模型会保留哪些 token。
7. 为什么当前 Tiny GPT 生成速度会随序列增长，并且没有自动停止能力。

## 1. 训练与生成的根本区别

训练时已经知道完整正确文本：

```text
input : 我 爱 学 习
target: 爱 学 习 。
```

因果 mask 保证每个位置看不到未来，因此一次 forward 可以并行计算 T 个合法预测：

```text
看到“我”       → 预测“爱”
看到“我爱”     → 预测“学”
看到“我爱学”   → 预测“习”
看到“我爱学习” → 预测“。”
```

生成时未来 token 不存在。模型必须：

```text
prompt → 预测 token₁
prompt+token₁ → 预测 token₂
prompt+token₁+token₂ → 预测 token₃
...
```

每次生成的 token 会成为下一轮真实输入，因此称为 autoregressive generation，自回归生成。

这里训练中的 `T` 是 sequence length，即输入序列中的 T 个位置，不是筛选出的“评分最高的 T 个候选”。每个位置都产生一份 V 维词表 logits，并用自己对应的 target 计算 loss。Top-k 是生成阶段对 V 个候选做的另一项操作，与序列长度 T 没有关系。

## 2. 项目的完整 generate()

```python
@torch.no_grad()
def generate(
    self,
    token_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
) -> Tensor:
    self.eval()
    for _ in range(max_new_tokens):
        context = token_ids[:, -self.config.block_size :]
        logits, _ = self(context)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            k = min(top_k, next_logits.size(-1))
            threshold = torch.topk(next_logits, k).values[:, -1, None]
            next_logits = next_logits.masked_fill(
                next_logits < threshold,
                float("-inf"),
            )

        probabilities = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids
```

主线：

```text
截取上下文
→ forward 得到 logits
→ 取最后位置
→ temperature 缩放
→ top-k 过滤
→ softmax
→ 随机采样 1 个 token
→ 拼回原序列
→ 重复
```

## 3. 一轮生成的 Shape

假设：

```text
B=1
当前上下文长度 T=12
词表 V=490
```

shape：

```text
token_ids                         (1,12)
context                           (1,12)
logits                            (1,12,490)
logits[:, -1, :]                  (1,490)
probabilities                     (1,490)
next_token                        (1,1)
cat(token_ids, next_token)        (1,13)
```

下一轮会使用长度 13 的序列，再生成第 14 个 token。

`max_new_tokens=80` 表示追加 80 个 token，不是让最终序列总长等于 80：

```text
最终长度 = prompt token 数 + max_new_tokens
```

例如 prompt 长度 10、`max_new_tokens=20`：

```text
最终长度 = 10+20 = 30
```

## 4. 为什么只取最后一个位置

forward 返回：

```text
logits.shape = (B,T,V)
```

其中每个位置的 logits 都是在预测它后面的 token。但 prompt 前面位置的“下一个 token”已经存在于输入中，不需要重新生成。

当前要追加到整个序列末尾的 token，只对应最后位置：

```python
next_logits = logits[:, -1, :]
```

shape 从 `(B,T,V)` 变成 `(B,V)`。

整数索引会消掉被索引的维度：

```text
logits[:, -1, :]   → (B,V)
logits[:, -1:, :]  → (B,1,V)
```

所以 `(2,12,490)` 经过 `logits[:, -1, :]` 后是 `(2,490)`，不是 `(2,1,490)`。

可以先用更小的 `(B,T,V)=(2,3,4)` 展开：

```text
logits = [
  [batch 0 的位置 0：4 个分数,
   batch 0 的位置 1：4 个分数,
   batch 0 的位置 2：4 个分数],

  [batch 1 的位置 0：4 个分数,
   batch 1 的位置 1：4 个分数,
   batch 1 的位置 2：4 个分数],
]
```

整个张量在训练时包含：

```text
B×T = 2×3 = 6 份 next-token prediction
每份 prediction 有 V=4 个候选分数
B×T×V = 2×3×4 = 24 个标量分数
```

这里的 `24` 是张量里的**分数总数**，不是 24 份预测。判断“有几份预测”时，先固定一个 `(b,t)` 位置；该位置对应的整条 `V` 维向量才是一份预测。

真实 PyTorch 索引结果：

```text
logits.shape          = (2,3,4)
logits[:, -1, :].shape  = (2,4)
logits[:, -1:, :].shape = (2,1,4)
```

`logits[:, -1, :]` 的具体含义是：保留两个 batch，只选择每个 batch 的最后位置，并保留该位置的全部 4 个候选分数。因此生成阶段这里仍有 B=2 份 next-token prediction，每个 batch 序列各生成一个新 token。

继续采样时：

```text
probabilities                         (2,4)
torch.multinomial(..., num_samples=1) (2,1)
```

`multinomial` 会从每行的 4 个候选概率中抽取 1 个 token ID，所以结果只保存“抽中了哪个 ID”，不再保留那 4 个候选分数。`(2,1,4)` 会错误地把候选维度继续保留下来。

“只取最后位置”也不表示忽略前面的上下文。经过因果 Attention 后，最后位置的隐藏状态已经汇总了它能够看到的前面 token 信息；对应的最后位置 logits 正是在“整个可见上下文之后，下一个 token 是什么”的评分。前面各位置 logits 预测的 next token 已经存在于 prompt 中，因此生成时不再使用。

## 5. Context window 与滑动窗口

项目默认：

```text
block_size = 64
```

每轮只保留最近 64 个 token：

```python
context = token_ids[:, -self.config.block_size :]
```

如果总序列还不到 64，模型读取全部内容；超过 64 后，最左侧的旧 token 会逐渐滑出上下文：

```text
完整输出仍保存在 token_ids 中
模型下一轮实际读取的只有最后 64 个 token
```

因此模型无法直接利用 64 个 token 之前的内容。这是 context window 的真实限制，不是生成结果被删除。

当前实现每轮把截取后的 context 重新编号为位置 `0...T-1`，这是小型教学实现的简化。

## 6. Temperature 做什么

项目先执行：

```python
scaled_logits = logits / temperature
```

然后 softmax。

对于：

```text
logits = [2,1,0]
```

有：

| temperature | 缩放后 logits | softmax 概率 |
| ---: | --- | --- |
| 0.5 | `[4,2,0]` | `[0.8668,0.1173,0.0159]` |
| 1.0 | `[2,1,0]` | `[0.6652,0.2447,0.0900]` |
| 2.0 | `[1,0.5,0]` | `[0.5065,0.3072,0.1863]` |

结论：

```text
0<T<1：差距被放大，分布更尖锐，更偏向最高分 token
T=1：保持原始 logits
T>1：差距被压缩，分布更平坦，低分 token 机会增大
```

只要 temperature 为正，所有 logits 都除以相同正数，因此排名不会改变；改变的是 softmax 后的概率差距。

更精确地说，`T<1` 会放大 logits 之间的差值，不等于每个 logit 数值都变大：正 logit 会变得更正，负 logit 会变得更负，0 仍是 0。softmax 因此更尖锐。在 `[2,1,0]` 例子中，最高分 token 的概率从 `0.6652` 升到 `0.8668`。

当 `T→0⁺` 时，最高 logit 的概率趋近于 1。项目使用：

```python
max(temperature, 1e-5)
```

避免除以 0。但调用者仍应该传入正温度；负温度在这里会被强行替换为 `1e-5`，不是有意义的采样配置。

## 7. Top-k 做什么

Top-k 先找到分数最高的 k 个 token，把其他 logits 改为负无穷：

```python
threshold = torch.topk(next_logits, k).values[:, -1, None]
next_logits = next_logits.masked_fill(
    next_logits < threshold,
    float("-inf"),
)
```

softmax 中：

```text
exp(-∞) = 0
```

所以被过滤 token 的最终概率严格为 0。

例如：

```text
logits = [2,1,0]
top_k  = 2
```

过滤后：

```text
[2,1,-∞]
```

在 `temperature=1` 时：

```text
probabilities ≈ [0.7311,0.2689,0]
```

Top-k 只限制候选池，不一定直接选择最高分 token：

```text
top_k=1：通常只剩最高分 token，接近 greedy decoding
top_k=20：只在最高的 20 个候选中采样
top_k>=V：相当于不过滤
top_k=None：模型方法中表示不过滤
```

当前 CLI 把 `--top-k` 定义为整数，不能直接从命令行传 `None`；传入不小于词表大小的值可以达到近似不过滤效果。

当前实现还要求启用 top-k 时 `top_k>=1`；传入 0 会因为无法取得第 k 个阈值而报错。生成 prompt 也应至少包含一个 token，否则没有“最后位置 logits”可取。

如果第 k 名附近存在完全相同的分数，阈值写法可能保留超过 k 个并列 token；普通浮点 logits 中这种精确并列并不常见。

## 8. Temperature 与 Top-k 的分工

```text
Temperature：调整候选之间的相对概率差距
Top-k：先删除排名低于前 k 的候选
```

两者组合顺序：

```text
raw logits
→ 除以 temperature
→ 保留 top-k
→ softmax
→ sampling
```

正 temperature 不改变排名，所以它不会改变哪些 token 进入 top-k；但它会改变 top-k 内部各 token 的采样概率。

典型现象：

| 配置倾向 | 常见效果 | 常见风险 |
| --- | --- | --- |
| 低 temperature、小 k | 稳定、保守 | 重复、模板化 |
| 中等 temperature、中等 k | 连贯性与多样性平衡 | 需要按模型调节 |
| 高 temperature、大 k | 多样、意外 | 语义破碎、低质量 token 增多 |

不存在适用于所有模型和任务的唯一最佳参数。

## 9. Sampling 与 Argmax

项目执行：

```python
next_token = torch.multinomial(probabilities, num_samples=1)
```

它根据概率随机抽样：概率高的 token 更容易被选中，但不保证每次都选概率最高的 token。

Greedy decoding 通常写作：

```python
next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
```

它每轮都选择最高分 token，结果通常确定，但容易进入重复模式，也不保证获得整段文本全局概率最高的序列。

项目没有直接实现 argmax；`top_k=1` 时通常只有最高分 token 概率非零，因此具有近似 greedy 的效果。若最高分精确并列，仍可能在并列项中随机选择。

## 10. 为什么同一 Prompt 会得到不同结果

当候选池中有多个非零概率 token 时，`torch.multinomial` 使用随机数采样。因此同一模型、同一 prompt、相同 temperature/top-k，也可能生成不同文本。

实验对比时可以先固定：

```python
torch.manual_seed(7)
```

然后一次只改一个参数。

随机种子只帮助复现实验，不会让采样变成模型学到的新规律；不同设备和 PyTorch 版本仍可能有细微差异。

## 11. 真实 Checkpoint 对照实验

使用同一个 1200-step checkpoint、prompt `模型学习`、随机种子 7，每组追加 80 个 token。

### Top-k=1：稳定但重复

```text
temperature=0.8, top_k=1

模型学习能力。
训练不断重复同一个模型如何处理输入模型如何处理输入。
训练不断重要先看到未来。
训练不断重复同一个小模型如何处理输入模型如何处理输入。
训练损失使用 t
```

最高分路径比较稳定，但很快出现“模型如何处理输入”等重复片段。

### 默认附近：较均衡

```text
temperature=0.8, top_k=20

模型学习参数字。
优化器完大型的内容。

训练时能不会容易多。
温度较低损失曲线了过索，然容易得到了，断重复同表示词表，准确的拟合。
它后的每个数字符组合程度易得更可以
```

比 top-k=1 多样，但 Tiny GPT 的数据和容量很小，仍不能期待大型模型的语言质量。

### 高温大候选池：更加发散

```text
temperature=1.4, top_k=100

模型学习语言模型生不能力。大模型的帮助批个络分新。语言留率越低。温序让也不获大损失使用没有强方向传播itopoi采样只保存模型拟合自同决换和预制，再计降字。

这据。
```

更多低分 token 得到机会，多样性增加，但连贯性明显下降。

这些输出不是证明某组参数永远更好，而是展示采样参数的方向性影响。

## 12. Eval 与 no_grad

方法装饰器：

```python
@torch.no_grad()
```

表示生成不记录反向传播计算图，减少内存和计算。

方法内部：

```python
self.eval()
```

关闭 Dropout，使同一组 logits 不再因训练模式的随机 Dropout 变化。

但只要仍使用 multinomial 采样，eval 模式不等于输出必然确定；采样本身仍然随机。

当前 `generate()` 会把模型留在 eval 模式，不会自动恢复之前的 train 模式。独立推理没有问题；如果在训练循环中间调用生成，之后需要显式 `model.train()` 才能恢复 Dropout。

## 13. 生成成本与 KV Cache

当前实现每生成一个 token 都重新计算整个最近上下文：

```text
第 1 轮：对 prompt forward
第 2 轮：对 prompt+token₁ 重新 forward
第 3 轮：对 prompt+token₁+token₂ 重新 forward
...
```

已经计算过的 token 的 K/V 也会重复计算。

现代 LLM 通常使用 KV Cache：保存过去 token 每层 Attention 的 Key/Value，新一轮主要计算新 token，再与缓存交互。这样显著降低自回归解码中的重复工作，但缓存会占用额外显存。

本项目故意不实现 KV Cache，以保持 Attention 与生成循环透明。

## 14. 当前实现没有 EOS 提前停止

很多 tokenizer 会定义 EOS（end-of-sequence）token。生成出 EOS 后可以提前停止。

当前字符级 tokenizer 没有专门 EOS，generate 也没有停止条件，因此它一定循环：

```text
max_new_tokens 次
```

输出长度只受 `max_new_tokens` 和外部调用控制，不会因为生成句号、换行或语义完成自动停止。

## 15. 生成中的错误会继续传播

训练时下一位置输入来自真实语料；生成时下一位置输入来自模型自己的采样。如果某一步选中了不合适的 token，它会成为后续上下文，影响之后所有预测。

```text
一次采样偏离
→ 新上下文改变
→ 后续概率分布改变
→ 可能继续偏离
```

采样参数只能控制选择策略，不能补回模型从未学到的知识，也不能把小模型自动变成大模型。

## 16. 其他常见采样方法

现代 LLM 还常用 top-p / nucleus sampling：保留累计概率达到 p 的最小候选集合。它的候选数量会随当前概率分布动态变化。

```text
top-k：固定最多考虑排名前 k 的候选
top-p：根据累计概率动态决定候选数量
```

本项目只实现 temperature 与 top-k，足以展示从 logits 到随机 token 的完整核心流程。

## 完整 Shape 与循环

```text
prompt IDs                                      (B,T₀)

循环第 i 次：
context = last block_size tokens               (B,T_context)
model(context)                                 (B,T_context,V)
last-position logits                           (B,V)
temperature + top-k + softmax                  (B,V)
multinomial                                    (B,1)
concatenate                                    (B,T₀+i)

最终输出                                       (B,T₀+max_new_tokens)
```

## 常见误区

1. temperature 不直接选择 token，它调整 softmax 概率差距。
2. 正 temperature 不改变 logit 排名，因此也不改变 top-k 候选集合。
3. top-k 不是“选择第 k 个 token”，而是只保留最高分的 k 个候选。
4. softmax 后仍要 sampling 或 argmax 才得到 token ID。
5. `model.eval()` 关闭 Dropout，但不会关闭 multinomial 的随机性。
6. `max_new_tokens` 是新增数量，不是最终总长度。
7. `token_ids` 保存完整输出，但模型每轮最多读取最后 `block_size` 个 token。
8. 当前实现没有 KV Cache，每轮会重复计算旧 token。
9. 当前实现没有 EOS，生成不会自行提前结束。
10. 提高 temperature 或 top-k 只能增加采样多样性，不能提高模型本身的知识和能力。
11. 训练中的 T 是序列位置数，不是 top-T 候选；每个位置都预测自己的下一个 token。
12. `logits[:, -1, :]` 使用整数索引，所以输出为 `(B,V)`；只有 `-1:` 切片才保留长度为 1 的位置维。
13. 最后位置 logits 已通过 Attention 包含可见上下文信息，并非只看最后一个 token 的孤立评分。
14. 单个 `logits[b,t,:]` 是一份 V 分类预测；完整 `(B,T,V)` 在训练时包含 `B×T` 份预测。
15. `B×T×V` 是标量分数总数，不是预测任务数；一次 V 分类预测由 V 个分数组成。
16. `multinomial((B,V), num_samples=1)` 为每个 batch 抽取一个 token ID，输出 shape 是 `(B,1)`。

## 验收题

1. 为什么训练能一次计算 T 个 next-token prediction，而生成通常必须一次生成一个 token？
2. `logits.shape=(2,12,490)` 时，`logits[:, -1, :]` 的 shape 是什么？为什么使用最后位置？
3. 对 `[2,1,0]`，temperature 从 1 降到 0.5 后，分布会更尖锐还是更平坦？最高分 token 的概率怎样变化？
4. `top_k=2` 后，第三名 token 的 logit 和概率分别变成什么？top-k 是否直接决定最终选中第一名？
5. prompt 有 10 个 token、`max_new_tokens=20`，最终长度是多少？若 `block_size=16`，最后一轮模型最多读取多少个 token？
6. `model.eval()` 后使用 `top_k=20` 和 multinomial，同一 prompt 是否保证每次输出相同？为什么？

## 验收结论

学员已经能够沿着完整生成循环追踪 shape：

```text
完整训练 logits (B,T,V)
→ 共 B×T 份 next-token prediction
→ 每份预测包含 V 个候选分数
→ 生成时取最后位置得到 (B,V)
→ multinomial 为每条序列采样一个 token ID，得到 (B,1)
→ 拼回 token_ids，进入下一轮自回归生成
```

对于 `(B,T,V)=(3,5,490)`，已经能正确判断：训练 forward 同时包含 `3×5=15` 份预测；`logits[:, -1, :]` 为 `(3,490)`；每条序列采样一个 token 后为 `(3,1)`。

还需要持续牢记两组区别：

```text
B×T：预测任务数
B×T×V：张量中的标量分数总数

-1：整数索引，删除被选中的位置维
-1:：切片，保留长度为 1 的位置维
```

至此，已经打通从 checkpoint 加载后的 logits，到 temperature、top-k、softmax、随机采样，再到逐 token 拼接输出的完整推理流程。
