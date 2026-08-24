# 第 1 课：Tokenizer 与训练样本

## 学习目标

完成本课后，应该能够解释：

1. 为什么语言模型不能直接读取文字。
2. `encode`、`decode`、`stoi` 和 `itos` 分别做什么。
3. Python 实例方法中的 `self` 是什么。
4. `torch.stack` 如何创建 batch 维度。
5. `make_batch` 如何生成输入 `x` 和目标 `y`。
6. 为什么目标必须相对输入右移一个 token。

## 1. 从文字到 token ID

本项目使用最简单的字符级 tokenizer，每个汉字、标点和换行都是一个 token：

```text
语言模型
↓
语  言  模  型
↓
427 415 308 164
```

编号只用于查表。`427` 不表示“语”的数值比 `415` 更大，也不包含加密功能。

```text
encode：文字 → token ID
decode：token ID → 文字
```

代码位于 `tinygpt.py` 的 `CharTokenizer`：

```python
def encode(self, text: str) -> List[int]:
    unknown = self.stoi[self.UNK]
    return [self.stoi.get(char, unknown) for char in text]

def decode(self, ids: List[int]) -> str:
    return "".join(...)
```

其中：

- `stoi` 是 string-to-integer，记录字符到整数的映射。
- `itos` 是 integer-to-string，记录整数到字符的映射。
- `<unk>` 表示词表里没有出现过的字符。

如果编码时遇到未知字符，它会变成 `<unk>` 的 ID；解码只能得到未知字符标记，因此这种情况下不保证完全还原原文。

## 2. `self` 表示当前对象

调用：

```python
tokenizer.encode("语言模型")
```

可以理解成 Python 自动执行：

```python
CharTokenizer.encode(tokenizer, "语言模型")
```

这里传入的 `tokenizer` 就成为方法内部的 `self`。因此：

- `self.stoi` 是当前 tokenizer 的词表。
- `self.itos` 是当前 tokenizer 的反向词表。
- 两个 tokenizer 实例可以拥有不同词表，`self` 用于区分当前操作哪个实例。

`self` 不是强制关键字，但它是 Python 的标准命名惯例。

## 3. 语言模型的训练目标

如果原始文本为：

```text
语言模型
```

训练样本会构造为：

```text
输入 x：语 言 模
目标 y：言 模 型
```

各位置同时训练：

| 模型已经看到 | 当前目标 |
| --- | --- |
| 语 | 言 |
| 语言 | 模 |
| 语言模 | 型 |

这叫作 next-token prediction，即根据左侧上下文预测下一个 token。

## 4. `torch.stack` 创建 batch 维度

假设有两个形状均为 `(3,)` 的 tensor：

```python
a = torch.tensor([10, 11, 12])
b = torch.tensor([20, 21, 22])
```

执行：

```python
torch.stack([a, b])
```

得到：

```text
[[10, 11, 12],
 [20, 21, 22]]
```

shape 从单个样本的 `(3,)` 变成 `(2,3)`。新增加的第一个维度就是 batch。

`stack` 与 `cat` 不同：

- `stack([a,b])` 新增维度，结果为 `(2,3)`。
- `cat([a,b])` 沿已有维度连接，结果为 `(6,)`。

参与 `stack` 的 tensor 必须拥有相同 shape。

## 5. 逐句理解 `make_batch`

函数签名：

```python
def make_batch(
    data: Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
```

参数含义：

- `data`：整篇语料编码后的一维 token ID tensor。
- `batch_size`：一次并行处理多少个文本片段，即 `B`。
- `block_size`：每个文本片段包含多少个 token，即 `T`。
- `device`：数据放在 CPU 或 Apple MPS 上计算。
- 返回值为输入 tensor `x` 和目标 tensor `y`。

计算合法起点数量：

```python
number_of_start_positions = len(data) - block_size
```

目标 `y` 会比输入多向右读取一个位置，所以长度为 `N` 的数据、长度为 `T` 的片段共有 `N-T` 个合法起点。

检查语料是否足够长：

```python
if number_of_start_positions <= 0:
    raise ValueError(...)
```

随机抽取 `B` 个起点：

```python
starts = torch.randint(number_of_start_positions, (batch_size,))
```

`torch.randint` 不包含上界，所以这里会得到从 `0` 到 `len(data)-block_size-1` 的起点。不同样本可以重叠，也可能碰巧选择相同起点。

构造输入：

```python
x = torch.stack([
    data[start : start + block_size]
    for start in starts
])
```

列表推导式等价于：

```python
pieces = []
for start in starts:
    pieces.append(data[start : start + block_size])
x = torch.stack(pieces)
```

构造右移一位的目标：

```python
y = torch.stack([
    data[start + 1 : start + block_size + 1]
    for start in starts
])
```

最后移动到计算设备：

```python
return x.to(device), y.to(device)
```

模型与数据必须位于同一设备，否则 PyTorch 会报告设备不一致。

## 6. 手算一个 batch

设定：

```text
data       = [10, 11, 12, 13, 14, 15, 16]
block_size = 3
batch_size = 2
starts     = [0, 2]
```

则：

```text
x = [[10, 11, 12],
     [12, 13, 14]]

y = [[11, 12, 13],
     [13, 14, 15]]
```

用文字表示：

```text
x 第 1 行：语 言 模    y 第 1 行：言 模 型
x 第 2 行：模 型 学    y 第 2 行：型 学 习
```

最终：

```text
x.shape = (B, T)
y.shape = (B, T)
```

虽然一整行同时输入模型，但后续因果遮罩会阻止前面的位置读取右侧答案。

## 7. 实验

运行一次前向检查：

```bash
.venv/bin/python inspect_forward.py
```

找到输出中的原始文本、token IDs 和 `(B,T)`，逐字符核对数量。

## 验收题

1. 编码与加密有什么区别？
2. `tokenizer.encode(text)` 中，哪个对象被自动传给 `self`？
3. 为什么目标 `y` 要比输入 `x` 右移一位？
4. 两个 `(8,)` tensor 经过 `torch.stack` 后是什么 shape？
5. `batch_size=4`、`block_size=16` 时，`x` 和 `y` 各是什么 shape？

能够脱离笔记回答以上问题，即可进入第 2 课。
