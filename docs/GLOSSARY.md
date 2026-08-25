# 术语与 Shape 速查表

## 核心术语

| 术语 | 含义 |
| --- | --- |
| token | 模型读取和预测的基本文本单位；本项目中是单个字符 |
| vocabulary / vocab | 所有可识别 token 的集合 |
| token ID | token 在词表中的整数编号，不代表数值大小或语义距离 |
| encode | 把文字转换成 token ID |
| decode | 把 token ID 转换回文字 |
| embedding | 把离散 ID 映射为可学习连续向量 |
| batch | 一次并行计算的一组训练样本 |
| context window | 模型一次最多读取的 token 数量 |
| next-token prediction | 根据左侧上下文预测下一个 token |
| parameter | 通过训练更新的模型内部数值 |
| gradient | loss 对参数的导数，指示参数变化如何影响 loss |
| logits | softmax 之前、针对每个候选 token 的原始分数 |
| query / Q | 当前位置为了读取上下文而生成的“查询”向量 |
| key / K | 每个位置用于和查询匹配的“索引”向量 |
| value / V | 匹配后真正被加权汇总的内容向量 |
| attention score | Q 与 K 点积得到的匹配分数，softmax 前还不是概率 |
| multi-head attention | 在多个特征子空间中并行计算注意力；每个 Head 都读取完整序列 |
| output projection | 合并 Head 后，用可学习线性层重新混合各 Head 的信息 |
| dropout | 训练时随机置零部分连接或特征的正则化方法，推理时关闭 |
| hidden state | 当前输入在模型各层之间传递的临时连续表示，本项目常见 shape 为 `(B,T,C)` |
| Transformer Block | 由 Attention、FFN、LayerNorm 和残差连接组成的基本层；多个 Block 按顺序堆叠 |
| LayerNorm | 对每个 token 的最后 C 维计算均值和方差并重新缩放，不等同于 Linear，也不保证正态分布 |
| residual connection | 将子层更新量加回原隐藏状态的连接，即 `x + F(x)` |
| FFN | 对每个 token 独立进行通道变换的前馈网络，通常为 `C → 4C → C` |
| GELU | FFN 中的非线性激活函数，使相邻 Linear 不能简单合并成一个 Linear |
| Pre-LN | 在进入 Attention 或 FFN 之前执行 LayerNorm 的结构，即 `x + F(LayerNorm(x))` |
| lm_head | 把最终隐藏状态从 C 维投影到词表 V 维，输出 logits；名称首字符是小写字母 l |

## Shape 字母

| 字母 | 常见含义 | 本项目示例 |
| --- | --- | ---: |
| B | batch size | 16 |
| T / S | time / sequence length | 64 |
| C / H | channel / hidden size | 64 |
| N | number of attention heads | 2 |
| D | head dimension，满足 `C=N×D` | 32 |
| V | vocabulary size | 490 |

## 常见 Shape

| 数据 | Shape |
| --- | --- |
| token IDs | `(B,T)` |
| token embedding | `(B,T,C)` 或 BSH |
| Q/K/V 多头表示 | `(B,N,T,D)` 或 BNSD |
| 注意力分数 | `(B,N,T,T)` 或 BNSS |
| 各 Head 加权输出 | `(B,N,T,D)` |
| 合并 Head 后 | `(B,T,C)` 或 BSH |
| logits | `(B,T,V)` |

看到不同字母时，优先检查代码或文档如何定义，不要假设所有项目都遵守同一命名。
