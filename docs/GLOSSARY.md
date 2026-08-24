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
| logits | `(B,T,V)` |

看到不同字母时，优先检查代码或文档如何定义，不要假设所有项目都遵守同一命名。
