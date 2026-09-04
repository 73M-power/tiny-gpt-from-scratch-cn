# 第 10 课：Checkpoint、模型加载与继续训练

> 状态：已完成。

## 学习目标

完成本课后，应该能够解释：

1. checkpoint 为什么不是“一个训练步数”，也不一定只有模型权重。
2. `model.state_dict()` 保存了什么，没有保存什么。
3. 为什么恢复推理至少需要模型参数、模型配置和 tokenizer 映射。
4. `torch.load()`、重新创建模型和 `load_state_dict()` 分别负责什么。
5. “加载后生成”“从旧权重继续训练”和“完整断点续训”的区别。
6. 为什么完整 resume 还需要 AdamW 状态、累计 step 和随机数状态。
7. 项目的 `--resume` 如何让训练步数累计，并恢复相同的训练轨迹。
8. `map_location="cpu"` 与 `weights_only=True` 的作用和边界。

## 1. Checkpoint 是什么

训练过程中有两类状态：

```text
内存中的临时状态：
当前 batch、hidden state、logits、loss、计算图、当前梯度

跨训练 step 持续存在的状态：
模型参数、optimizer 历史、累计 step、随机数生成器状态、配置、tokenizer
```

程序退出后，内存中的内容都会消失。checkpoint 是把以后需要恢复的持久状态序列化到磁盘的快照。

它可以类比为游戏存档：

```text
只有角色外观             → 不足以恢复游戏
角色属性 + 任务进度 + 地图 → 更接近完整存档
```

同样：

```text
只有 model_state          → 可以恢复模型参数，但不一定能恢复完整训练
完整 training checkpoint → 还包含 optimizer、step、RNG、结构和词表等状态
```

因此 checkpoint 描述的是“保存下来的状态集合”，`.pt` 只是本项目使用的文件扩展名。

还要区分“有一个 checkpoint 文件”和“checkpoint 内真的保存了某项状态”。例如文件中若只有：

```python
{"training_steps": 1200}
```

它只能告诉我们曾记录到第 1200 步，不能恢复任何权重。恢复模型权重必须实际包含 `model_state`；恢复完整训练还要继续包含 optimizer 和 RNG 等状态。

## 2. 当前已有 checkpoint 的真实内容

检查旧文件：

```python
checkpoint = torch.load(
    "checkpoints/tinygpt.pt",
    map_location="cpu",
    weights_only=True,
)
print(checkpoint.keys())
```

实测结果：

```text
model_state
config
tokenizer_itos
training_steps
```

文件约为 `590,345` bytes，也就是约 `576.5 KiB`。其中：

| 键 | 内容 | 用途 |
| --- | --- | --- |
| `model_state` | 模型参数和持久 buffer | 恢复已经学到的数值 |
| `config` | 层数、Head 数、隐藏维度、词表大小等 | 重新创建相同结构 |
| `tokenizer_itos` | token ID 到字符的原始顺序 | 恢复完全相同的编码语义 |
| `training_steps` | 保存时记录的累计训练步数 | 显示进度；它本身不会让循环自动跳到该步 |

这个旧文件可以继续用于 `generate.py`，但它没有 `optimizer_state` 和 `rng_state`，因此不能被当作完整训练状态执行 `--resume`。

## 3. `state_dict()` 不等于整个 Python 模型

`model.state_dict()` 返回一个从名称到 Tensor 的有序映射。例如：

```text
token_embedding.weight                       (490,64)
position_embedding.weight                    (64,64)
blocks.0.ln_1.weight                         (64,)
blocks.0.attention.query.weight              (64,64)
blocks.0.attention.causal_mask               (1,1,64,64)
...
```

它保存两类 Tensor：

```text
Parameter：训练中由 optimizer 更新的可学习参数
Buffer：属于模型状态、但不由 optimizer 更新的持久 Tensor
```

本项目的 `causal_mask` 通过：

```python
self.register_buffer("causal_mask", mask)
```

注册，因此会进入 `state_dict()`，但不会进入 `model.parameters()`，AdamW 也不会更新它。

`causal_mask` 是因果注意力的“可见范围表”。若序列长度 `T=4`，它是一个下三角矩阵：

```text
              可以读取的 key 位置
              0  1  2  3
query 位置 0 [1, 0, 0, 0]
query 位置 1 [1, 1, 0, 0]
query 位置 2 [1, 1, 1, 0]
query 位置 3 [1, 1, 1, 1]
```

矩阵中的含义是：

```text
1：当前位置可以关注这个位置
0：这个位置属于未来，必须遮住
```

例如 query 位置 2 可以读取位置 `0、1、2`，不能读取未来的位置 3。代码把被遮住位置的 attention score 改为负无穷；经过 softmax 后，它们的注意力概率变成 0。

实际 mask 的 shape 是 `(1,1,block_size,block_size)`，前两个 `1` 可以广播到所有 batch 和所有 Head。它由模型结构和 `block_size` 决定，不需要通过训练学习，所以属于 Buffer：

```text
需要跟着模型保存、加载、移动设备
不需要梯度
不由 AdamW 修改
```

本模型有 `135,040` 个可训练参数。旧 checkpoint 的 `model_state` 有 31 个命名 Tensor 条目；条目元素数不能直接当作“独立参数数”，原因包括：

- `state_dict()` 还包含 causal mask buffer。
- `token_embedding.weight` 与 `lm_head.weight` 是两个名称，但共享同一份底层权重存储。

`state_dict()` 保存的是数值状态，不包含 `TinyGPT.forward()` 的 Python 实现。加载时仍然需要当前代码中的 `TinyGPT` 类。

## 4. 为什么还必须保存 Config

只有一堆 Tensor，还不知道应该创建怎样的容器来接收它们。

本项目保存：

```python
config.to_dict()
```

真实配置为：

```python
{
    "vocab_size": 490,
    "block_size": 64,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "dropout": 0.1,
}
```

加载时先执行：

```python
config = GPTConfig(**checkpoint["config"])
model = TinyGPT(config)
```

这一步创建的是结构正确、但参数仍为随机初始化值的新模型。随后：

```python
model.load_state_dict(checkpoint["model_state"])
```

才把训练好的数值装入对应名称和 shape 的位置。

如果错误地用 `n_embd=128` 创建模型，而 checkpoint 中的矩阵是按 `n_embd=64` 保存的，`load_state_dict()` 会报告 size mismatch。它不能自动猜测怎样把 `(64,64)` 权重变成 `(128,128)`。

## 5. 为什么必须原样恢复 Tokenizer

模型只看到整数 ID。假设训练时：

```text
ID 10 → “模”
ID 11 → “型”
```

如果加载时重新排序成：

```text
ID 10 → “型”
ID 11 → “模”
```

即使词表大小仍然是 490，embedding 的每一行也会对应错误字符。模型参数没有损坏，但输入和输出的含义被换掉了。

因此项目保存 `tokenizer.itos`，并使用：

```python
tokenizer = CharTokenizer.from_itos(checkpoint["tokenizer_itos"])
```

按原顺序重建 `itos` 和 `stoi`。这里重要的不是“有哪些字符”，而是“每个字符对应哪个 ID”。

## 6. 完整推理加载链路

`generate.py` 的核心顺序是：

```python
checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True,
)
config = GPTConfig(**checkpoint["config"])
tokenizer = CharTokenizer.from_itos(checkpoint["tokenizer_itos"])
model = TinyGPT(config)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
```

数据流是：

```text
.pt 文件
→ torch.load 还原为 Python dict 和 Tensor
→ config 决定模型结构
→ tokenizer 恢复 ID 语义
→ TinyGPT(config) 创建参数容器
→ load_state_dict 填入训练后的数值
→ model.to(device) 移动到 CPU 或 MPS
→ model.generate 进行自回归生成
```

`torch.load()` 不会自动替你调用 `TinyGPT()`，`load_state_dict()` 也不会创建模型结构。两者职责不同。

## 7. 三种容易混淆的“恢复”

| 操作 | 恢复模型参数 | 恢复 optimizer | 恢复 RNG/step | 结果 |
| --- | --- | --- | --- | --- |
| 加载后推理 | 是 | 不需要 | 不需要 | 可以生成文本 |
| warm start / 从权重再训练 | 是 | 否，创建新 optimizer | 通常否 | 从已有参数出发，但训练轨迹改变 |
| resume / 断点续训 | 是 | 是 | 是 | 尽可能从保存点继续原训练过程 |

只执行：

```python
model.load_state_dict(checkpoint["model_state"])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

不是完整 resume，因为新的 AdamW 没有过去的历史统计。它属于 warm start。

## 8. 为什么 AdamW 状态必须保存

第 7 课已经学习过，AdamW 为每个参数维护：

```text
exp_avg     ：梯度一阶移动平均，可理解为带惯性的更新方向 m
exp_avg_sq  ：梯度平方的移动平均，可理解为更新尺度 v
step        ：该参数已经进行过多少次 optimizer 更新
```

这些内容由：

```python
optimizer.state_dict()
```

获得。模型参数相同，但 AdamW 历史不同，下一次 `optimizer.step()` 的结果也可能不同。

因此完整训练 checkpoint 新增：

```text
optimizer_state
```

加载时必须先创建绑定到新模型参数的 AdamW，再把保存的状态装进去：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=saved_lr)
optimizer.load_state_dict(checkpoint["optimizer_state"])
```

optimizer 内部的 Tensor 也必须和模型位于兼容设备上。

## 9. 为什么还要保存随机状态

本项目中的随机性至少影响：

- `make_batch()` 抽取训练窗口的位置。
- 训练模式下 Dropout 丢弃哪些激活。
- 生成阶段 multinomial 抽中哪个 token。

随机数生成器不是每次都从头独立抽签；它有一个会不断向前推进的内部状态。

因此新 checkpoint 保存：

```python
rng_state = {
    "python": random.getstate(),
    "torch_cpu": torch.get_rng_state(),
    "device_type": device.type,
}
```

使用 MPS 时还保存 `torch_mps`。严格 resume 要求当前训练设备类型与 checkpoint 的 `device_type` 相同；CPU 与 MPS 的随机流和数值算法不同，静默切换设备不能再声称训练轨迹被完整恢复。

恢复工作必须放在重新创建模型和 optimizer 之后，因为模型随机初始化本身会消耗随机数；最后恢复 RNG，下一次随机操作才能从保存点继续。

## 10. 一个真实发现：评估也会影响训练随机序列

`estimate_loss()` 会调用 `make_batch()` 随机抽取评估窗口。原实现和训练共用全局 Torch RNG，因此：

```text
多运行一次评估
→ RNG 向前推进
→ 下一训练 batch 的起点改变
→ 后续梯度和参数改变
```

分段训练通常会在边界多运行一次评估。如果只保存和恢复 RNG、却让评估继续修改它，恢复后的训练轨迹仍可能不同。

现在 `estimate_loss()` 在评估前保存 RNG，评估结束后再恢复：

```text
保存训练 RNG
→ 随机抽取验证 batch
→ 计算平均 loss
→ 恢复训练 RNG
→ 下一训练 batch 不受评估次数影响
```

这不是让验证结果不再随机，而是隔离“评估随机流”和“训练随机流”的副作用。

## 11. 新版完整训练 Checkpoint

更新后的 `save_training_checkpoint()` 保存：

```python
{
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "config": config.to_dict(),
    "tokenizer_itos": tokenizer.itos,
    "training_steps": training_steps,
    "rng_state": rng_state,
}
```

六项可以分为：

```text
模型本体：model_state + config
文本语义：tokenizer_itos
训练进度：optimizer_state + training_steps + rng_state
```

optimizer 通常还会保存与参数规模相近的 `exp_avg` 和 `exp_avg_sq`，所以完整训练 checkpoint 往往明显大于只用于推理的权重文件。这是正常代价。

## 12. `--resume` 的运行语义

先创建一个新格式 checkpoint：

```bash
.venv/bin/python train.py \
  --steps 30 \
  --eval-interval 10 \
  --eval-iters 3 \
  --output checkpoints/demo-step-30.pt
```

再恢复并新增 20 个训练 step：

```bash
.venv/bin/python train.py \
  --resume checkpoints/demo-step-30.pt \
  --steps 20 \
  --output checkpoints/demo-step-50.pt
```

这里 `--steps` 表示本次运行新增的更新次数：

```text
checkpoint 已完成 30 steps
本次 --steps 20
新 checkpoint training_steps = 30+20 = 50
```

`--steps` 必须大于或等于 0。`--output` 还必须与 `--resume` 指向不同文件，否则程序会在训练前拒绝执行，避免把唯一可用恢复点原地覆盖。

恢复时，模型结构、tokenizer 和 learning rate 来自 checkpoint；命令行中的 `--n-layer`、`--n-head`、`--n-embd`、`--block-size`、`--dropout` 和 `--learning-rate` 不会替换保存状态。

训练数据文件仍由 `--data` 指定。若希望延续同一个实验，应继续使用相同数据；当前 checkpoint 尚未保存数据文件或数据哈希。

## 13. 连续训练与 Resume 对照实验

CPU 上使用相同代码、数据、配置和随机种子，进行两组实验：

```text
实验 A：连续训练 3 steps
实验 B：训练 1 step → 保存 → resume 2 steps
```

比较最终 `model_state`：

```text
all_equal        True
max_abs_diff     0.0
different_tensors 0
training_steps   3 / 3
```

这说明在该受控 CPU 实验中，模型参数、AdamW 状态、RNG 和评估随机副作用都已正确恢复。

这项结果同时由 CLI 集成测试保护：测试故意给 resume 命令传入冲突的模型结构与 learning rate，最终仍逐 Tensor 对比完整模型状态、全部 AdamW 状态和 RNG。

“可恢复完整状态”不等于任何环境都保证逐 bit 一致。当前严格 resume 会拒绝 CPU/MPS 设备类型切换；即使设备类型相同，更换 PyTorch 版本、具体硬件、并行算法或数据仍可能产生差异，因此真实训练还会记录软件环境、代码提交和数据版本。

## 14. `map_location="cpu"` 做什么

checkpoint 可能是在 MPS、CUDA 或 CPU 上保存的。加载时先指定：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

表示先把 Tensor 还原到 CPU，再由代码明确执行：

```python
model.to(device)
```

这样加载逻辑不依赖保存机器是否具有相同加速设备，也更容易检查 checkpoint 内容。

恢复 optimizer 后，还要把它内部的 `exp_avg`、`exp_avg_sq` 等 Tensor 移到目标设备；只移动模型参数是不够的。

## 15. `weights_only=True` 做什么

PyTorch checkpoint 使用序列化格式。`weights_only=True` 限制反序列化器只接受权重 checkpoint 常见的安全数据类型，降低加载任意 Python 对象的风险。

它不代表互联网上的陌生文件可以无条件信任。仍应优先加载自己生成或来源可信、可校验的 checkpoint。

## 16. `load_state_dict()` 默认严格匹配

项目调用：

```python
model.load_state_dict(checkpoint["model_state"])
```

默认 `strict=True`，要求 checkpoint 和当前模型的参数/buffer 名称匹配。常见失败包括：

```text
missing keys     当前模型需要，但 checkpoint 没有
unexpected keys checkpoint 中存在，但当前模型不认识
size mismatch    名称相同，但 Tensor shape 不同
```

在教学项目中，严格失败通常比静默跳过更好，因为它会及时暴露“代码结构与权重不兼容”。

## 17. 当前 Resume 的边界

项目已经恢复：

- 模型 parameters 和 persistent buffers。
- AdamW 状态与保存的 learning rate。
- 模型 config 和 tokenizer ID 顺序。
- 累计 training step。
- Python、Torch CPU，以及使用时的 MPS RNG 状态。
- 保存时的训练设备类型；严格 resume 不允许静默切换 CPU/MPS。

仍未保存：

- 训练数据本身或内容哈希。
- Python/PyTorch 版本、硬件和代码提交哈希。
- train/val loss 历史与 best-val checkpoint。
- learning-rate scheduler 状态，因为当前项目没有 scheduler。
- 每隔若干 step 的自动 checkpoint；当前仍只在本次训练结束时保存。
- 临时文件写完再原子替换的防损坏机制。

这些是把教学脚本扩展为正式训练系统时需要继续补充的工程能力。

## 完整状态流

```text
首次训练：
data
→ tokenizer + config
→ TinyGPT + AdamW
→ forward/backward/step
→ model_state + optimizer_state + step + RNG
→ torch.save(checkpoint)

恢复训练：
torch.load(checkpoint on CPU)
→ checkpoint config 创建 TinyGPT
→ load model_state
→ 创建 AdamW 并 load optimizer_state
→ 移动模型和 optimizer Tensor 到 device
→ 恢复 tokenizer、累计 step 与 RNG
→ 新增 N 个训练 step
→ 保存累计后的新 checkpoint
```

## 常见误区

1. checkpoint 不是训练 step 数字；step 只是 checkpoint 中的一项元数据。
2. `.pt` 不是固定格式；里面有哪些键由保存代码决定。
3. `state_dict()` 不包含 `forward()` 代码，也不等于完整 Python 模型对象。
4. 相同词表大小不代表 tokenizer 相同，token ID 顺序也必须一致。
5. `TinyGPT(config)` 只创建结构和随机参数，`load_state_dict()` 才装入训练结果。
6. 只加载 `model_state` 再创建新 AdamW 属于 warm start，不是完整 resume。
7. optimizer 状态不属于模型 parameter，但会影响下一次参数更新。
8. `training_steps=1200` 不会让 Python 循环自动从 1200 开始，代码必须显式读取和累计。
9. RNG 不只是“seed 数字”；训练中真正要恢复的是随机数生成器当前状态。
10. 评估如果共享并推进训练 RNG，也可能间接改变后续训练轨迹。
11. `map_location="cpu"` 是加载位置，不代表模型最终只能在 CPU 运行。
12. 完整训练 checkpoint 通常比推理 checkpoint 大，因为 optimizer 还保存历史 Tensor。
13. 旧版 `tinygpt.pt` 仍能生成，但缺少 optimizer/RNG，不能严格 resume。
14. `--steps` 是非负的新增步数；负数不能表示“退回以前的模型”。
15. `--output` 不能和 `--resume` 是同一个文件，否则会覆盖恢复来源。
16. 跨 CPU/MPS 加载权重可以作为迁移思路，但不属于本项目保证训练轨迹的严格 resume。

## 验收题

1. `TinyGPT(config)` 和 `model.load_state_dict(...)` 分别做什么？
2. 为什么只保存 `model_state` 可以用于推理，却不能完整恢复 AdamW 训练？
3. `config` 中的 `n_embd` 与 checkpoint 权重 shape 不一致时会发生什么？
4. 两个 tokenizer 都有 490 个 token，但 ID 顺序不同，模型还能正确理解原来的 ID 吗？为什么？
5. checkpoint 已完成 1200 steps，运行 `--resume ... --steps 200` 后，新 checkpoint 的 `training_steps` 应是多少？
6. 为什么恢复 RNG 应放在重新创建模型和 optimizer 之后？
7. `model_state` 中为什么会出现 `causal_mask`？AdamW 会更新它吗？
8. 连续训练与分段恢复时，为什么额外运行评估也可能造成最终参数不同？项目如何隔离这个影响？
9. 为什么严格 resume 要拒绝把 CPU checkpoint 静默切换到 MPS？

## 验收结论

学习者已经能够说明：

- `TinyGPT(config)` 先创建结构和随机参数，`load_state_dict()` 再装入训练好的数值。
- 相同词表大小不代表 token ID 含义相同，恢复时必须保持 tokenizer 顺序一致。
- `causal_mask` 是固定规则对应的 Buffer，不是 AdamW 更新的 Parameter。
- 完整 resume 需要模型、optimizer、累计 step 和 RNG；只加载模型并创建新 optimizer 属于 warm start。
- AdamW 的历史状态绑定具体 Parameter，会影响下一次更新方向和尺度。
- 重新设置 seed 会回到随机序列起点，恢复 RNG state 才会回到保存位置。
- 验证中的随机采样会推进 RNG，因此验证前后需要保存并恢复训练 RNG。
- 恢复来源与新输出使用不同路径，可以保留上一份可恢复 checkpoint。
- `map_location="cpu"` 只是控制初始加载位置，不限制模型最终运行设备。
- `weights_only=True` 降低反序列化风险，但不能保证陌生 checkpoint 绝对安全。
- 严格加载能够区分 missing key、unexpected key 与 size mismatch。
- optimizer 的状态 Tensor 必须与模型参数位于兼容设备，否则更新时会发生设备不匹配错误。
