"""用一个很小的 batch 查看一次前向传播的关键形状和因果注意力。"""

from pathlib import Path

import torch

from tinygpt import CharTokenizer, GPTConfig, TinyGPT


def main() -> None:
    text = Path("data/tiny_chinese.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    sample = "语言模型学习下一个字符。"
    token_ids = torch.tensor([tokenizer.encode(sample)], dtype=torch.long)
    targets = torch.roll(token_ids, shifts=-1, dims=1)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = TinyGPT(config)
    logits, loss = model(token_ids, targets, capture_attention=True)
    attention = model.blocks[0].attention.last_attention

    print(f"原始文本           : {sample}")
    print(f"token IDs          : {token_ids.tolist()}")
    print(f"输入形状 (B, T)    : {tuple(token_ids.shape)}")
    print(f"logits (B, T, V)   : {tuple(logits.shape)}")
    print(f"attention (B,H,T,T): {tuple(attention.shape)}")
    print(f"随机初始化 loss    : {loss.item():.4f}")
    print("\n第 1 个注意力头的矩阵（每行只能关注本行及左侧）：")
    print(torch.round(attention[0, 0] * 100) / 100)


if __name__ == "__main__":
    main()
