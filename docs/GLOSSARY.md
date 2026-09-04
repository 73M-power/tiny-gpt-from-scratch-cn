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
| training step | 使用一个 batch 完成一次 forward、backward 和参数更新 |
| epoch | 通常指完整遍历训练集一次；随机有放回窗口采样时没有严格 epoch 边界 |
| training set | 用于计算梯度并更新模型参数的数据 |
| validation set | 不参与参数更新，用于训练期间评估、选择模型和调整超参数的数据 |
| test set | 模型与超参数确定后，用于最终独立报告的数据 |
| generalization | 模型把学到的规律迁移到未参与训练的数据上的能力 |
| overfitting | 训练表现继续改善，但验证表现停滞或变差的现象 |
| data leakage | 验证或测试信息不当进入训练过程，使评估结果失真的问题 |
| eval_interval | 相隔多少个训练 step 执行一次评估 |
| eval_iters | 一次评估平均多少个随机 batch；影响估计稳定性和评估成本，不影响更新步幅 |
| checkpoint | 某个训练 step 保存到磁盘的状态快照，可包含模型参数、optimizer 状态、配置等 |
| state_dict | PyTorch 用名称映射到 Tensor 的状态字典；模型的 `state_dict` 保存参数和持久化 Buffer，不包含 `forward()` 代码 |
| buffer | 属于模型状态但不由 optimizer 学习的 Tensor，例如固定的 `causal_mask`；可随模型移动设备并进入 `state_dict` |
| warm start | 加载已有模型权重，但重新创建 optimizer 等训练状态后开始新的训练轨迹 |
| resume | 从 checkpoint 恢复模型、optimizer、累计 step 和 RNG 等状态，继续原训练过程 |
| RNG state | 随机数生成器当前所在位置的完整状态；恢复它比仅重新设置初始 seed 更适合续训 |
| seed | 随机序列的起始种子；重新设置 seed 会回到序列起点，而恢复 RNG state 会回到保存时的位置 |
| map_location | `torch.load` 指定 Tensor 初始加载位置的参数；设为 CPU 不限制模型之后移动到其他设备 |
| weights_only | `torch.load` 的受限反序列化模式，只允许权重 checkpoint 常见数据类型以降低风险，但不保证陌生文件绝对安全 |
| strict loading | `load_state_dict` 默认的严格匹配方式，会报告 missing key、unexpected key 和 size mismatch |
| early stopping | 验证指标长期不再改善时停止训练，避免继续过拟合 |
| prompt | 交给模型作为生成起点的输入 token 序列 |
| autoregressive generation | 每轮预测一个新 token，把它拼回输入，再用扩展后的序列继续预测 |
| decoding / generation strategy | 从 logits 或概率分布选择下一个 token 的规则，例如 greedy、temperature、top-k 或 top-p |
| max_new_tokens | 最多新增的 token 数；不是包含 prompt 的最终总长度 |
| context window | 模型一次最多读取的 token 数量；生成结果可更长，但下一轮只能直接使用窗口内的最近 token |
| temperature | 生成时用 logits 除以正温度值来调节概率差距；较低时更集中，较高时更平坦 |
| top-k sampling | 只保留 logits 最高的 k 个候选，其余设为负无穷后再采样 |
| top-p / nucleus sampling | 保留累计概率达到阈值 p 的最小候选集合，候选数量会随分布改变 |
| sampling | 按概率随机抽取 token，因此同一概率分布可能得到不同结果 |
| multinomial | PyTorch 按每行概率采样类别索引的操作；输入 `(B,V)`、每行采 1 个时输出 `(B,1)` |
| greedy decoding | 每轮使用 argmax 选择最高分 token；通常稳定但更容易重复 |
| argmax | 返回最大值所在的索引；与随机采样不同，相同输入下选择确定 |
| EOS | end-of-sequence，序列结束 token；生成到它时可以提前停止，当前项目未定义 |
| KV Cache | 缓存已处理 token 的 Attention Key/Value，避免生成每个新 token 时重复计算全部旧位置；当前项目未实现 |
| next-token prediction | 根据左侧上下文预测下一个 token |
| parameter | 通过训练更新的模型内部数值 |
| gradient | loss 对参数的局部导数，指示当前点参数变化如何影响 loss；通常与对应参数 shape 相同 |
| learning rate / lr | 控制每次参数更新基础步幅的超参数 |
| backpropagation / backward | 从标量 loss 反向应用链式法则，把梯度累积到各参数 `.grad` 的过程 |
| autograd | PyTorch 自动记录张量运算并计算梯度的系统 |
| optimizer | 读取参数梯度并按照更新规则修改模型参数、维护优化器状态的对象 |
| zero_grad | 清理 optimizer 管理的参数梯度，不会清零模型参数 |
| gradient clipping | 梯度范数过大时按比例缩小梯度，降低单次更新失控的风险 |
| AdamW | 使用梯度一阶、二阶移动平均并采用解耦 weight decay 的优化器 |
| optimizer state | AdamW 为参数维护的 m/v 等跨 step 状态，不属于 forward 隐藏状态 |
| logits | softmax 之前、针对每个候选 token 的原始分数 |
| softmax | 沿候选类别维把 logits 转换成总和为 1 的概率分布 |
| target | 正确 token 的整数词表索引；单个位置是一个 ID，整批 targets 的 shape 为 `(B,T)` |
| cross-entropy loss | 正确 token 概率的负对数；项目默认再对 `B×T` 个位置取平均 |
| weight tying | 让 token embedding 与 `lm_head` 共享同一个 `(V,C)` 参数矩阵 |
| query / Q | 当前位置为了读取上下文而生成的“查询”向量 |
| key / K | 每个位置用于和查询匹配的“索引”向量 |
| value / V | 匹配后真正被加权汇总的内容向量 |
| attention score | Q 与 K 点积得到的匹配分数，softmax 前还不是概率 |
| multi-head attention | 在多个特征子空间中并行计算注意力；每个 Head 都读取完整序列 |
| output projection | 合并 Head 后，用可学习线性层重新混合各 Head 的信息 |
| dropout | 训练 forward 时随机置零部分中间激活或注意力连接的正则化方法；不删除参数，评估与推理时关闭 |
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
| softmax probabilities | `(B,T,V)` |
| 训练 logits 中的预测任务数 | `B×T`，每份预测包含 `V` 个分数 |
| 生成时最后位置 logits / probabilities | `(B,V)` |
| 每轮采样得到的 next token IDs | `(B,1)` |
| 生成完成后的 token IDs | `(B,T₀+max_new_tokens)` |
| targets | `(B,T)` |
| 展平后的 logits / targets | `(B×T,V)` / `(B×T)` |
| cross-entropy loss | 标量 `()` |
| 模型参数 / 参数梯度 | shape 相同，例如 `(V,C)` / `(V,C)` |
| AdamW 的 m/v 状态 | 通常与对应参数 shape 相同 |

看到不同字母时，优先检查代码或文档如何定义，不要假设所有项目都遵守同一命名。
