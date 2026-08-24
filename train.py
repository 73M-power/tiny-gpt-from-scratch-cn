"""训练 Tiny GPT，并把模型、配置和 tokenizer 一起保存。"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import time
from typing import Dict, Tuple

import torch
from torch import Tensor

from tinygpt import CharTokenizer, GPTConfig, TinyGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从零训练一个字符级 Tiny GPT")
    parser.add_argument("--data", type=Path, default=Path("data/tiny_chinese.txt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tinygpt.pt"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("当前 PyTorch 无法使用 MPS")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_batch(
    data: Tensor, batch_size: int, block_size: int, device: torch.device
) -> Tuple[Tensor, Tensor]:
    number_of_start_positions = len(data) - block_size
    if number_of_start_positions <= 0:
        raise ValueError("语料太短，请减小 block_size 或增加文本")
    # randint 的上界不包含在结果中，因此会采样 0 到 len(data)-block_size-1。
    starts = torch.randint(number_of_start_positions, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    splits: Dict[str, Tensor],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    result = {}
    for split_name, split_data in splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = make_batch(split_data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[split_name] = sum(losses) / len(losses)
    model.train()
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = int(0.9 * len(encoded))
    splits = {"train": encoded[:split_at], "val": encoded[split_at:]}

    if len(splits["val"]) <= args.block_size + 1:
        raise ValueError("验证集太短，请减小 block_size 或增加文本")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"device       : {device}")
    print(f"characters   : {len(text):,}")
    print(f"vocab size   : {tokenizer.vocab_size}")
    print(f"parameters   : {model.count_parameters():,}")
    print(f"context      : {config.block_size} characters")

    started = time.perf_counter()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(
                model,
                splits,
                args.batch_size,
                config.block_size,
                args.eval_iters,
                device,
            )
            print(
                f"step {step:4d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f}"
            )
        if step == args.steps:
            break

        inputs, targets = make_batch(
            splits["train"], args.batch_size, config.block_size, device
        )
        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer_itos": tokenizer.itos,
            "training_steps": args.steps,
        },
        args.output,
    )
    elapsed = time.perf_counter() - started
    print(f"saved        : {args.output}")
    print(f"elapsed      : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
