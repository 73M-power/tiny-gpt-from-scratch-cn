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
    parser = argparse.ArgumentParser(description="训练或恢复一个字符级 Tiny GPT")
    parser.add_argument("--data", type=Path, default=Path("data/tiny_chinese.txt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tinygpt.pt"))
    parser.add_argument("--resume", type=Path, help="从完整训练 checkpoint 恢复")
    parser.add_argument("--steps", type=int, default=1200, help="本次新增的训练步数")
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
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps 必须大于或等于 0")
    if args.resume is not None and args.output.resolve() == args.resume.resolve():
        parser.error("--output 不能与 --resume 指向同一个 checkpoint")
    return args


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
    torch_rng_state = torch.get_rng_state()
    mps_rng_state = torch.mps.get_rng_state() if device.type == "mps" else None
    model.eval()
    try:
        result = {}
        for split_name, split_data in splits.items():
            losses = []
            for _ in range(eval_iters):
                x, y = make_batch(split_data, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[split_name] = sum(losses) / len(losses)
    finally:
        torch.set_rng_state(torch_rng_state)
        if mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)
        model.train()
    return result


def save_training_checkpoint(
    path: Path,
    model: TinyGPT,
    optimizer: torch.optim.AdamW,
    config: GPTConfig,
    tokenizer: CharTokenizer,
    training_steps: int,
    device: torch.device,
) -> None:
    """保存能够恢复模型与 AdamW 历史的训练状态。"""
    rng_state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "device_type": device.type,
    }
    if device.type == "mps":
        rng_state["torch_mps"] = torch.mps.get_rng_state()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "tokenizer_itos": tokenizer.itos,
            "training_steps": training_steps,
            "rng_state": rng_state,
        },
        path,
    )


def load_training_checkpoint(
    path: Path, device: torch.device
) -> Tuple[TinyGPT, torch.optim.AdamW, GPTConfig, CharTokenizer, int]:
    """从 checkpoint 重建模型、optimizer、配置、tokenizer 与累计 step。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = {
        "model_state",
        "optimizer_state",
        "config",
        "tokenizer_itos",
        "training_steps",
        "rng_state",
    }
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"checkpoint 缺少完整恢复训练所需字段：{missing}。"
            "它仍可能用于推理，但不能用于 --resume。"
        )

    rng_state = checkpoint["rng_state"]
    required_rng_keys = {"python", "torch_cpu", "device_type"}
    if not isinstance(rng_state, dict):
        raise ValueError("checkpoint 的 rng_state 必须是字典")
    missing_rng_keys = sorted(required_rng_keys.difference(rng_state))
    if missing_rng_keys:
        missing = ", ".join(missing_rng_keys)
        raise ValueError(f"checkpoint 的 rng_state 缺少字段：{missing}")

    saved_device_type = rng_state["device_type"]
    if saved_device_type not in {"cpu", "mps"}:
        raise ValueError(f"checkpoint 包含不支持的训练设备：{saved_device_type}")
    if saved_device_type == "mps" and "torch_mps" not in rng_state:
        raise ValueError("checkpoint 的 rng_state 缺少字段：torch_mps")
    if saved_device_type != device.type:
        raise ValueError(
            f"严格 resume 要求相同设备类型：checkpoint={saved_device_type}, "
            f"current={device.type}"
        )

    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer.from_itos(checkpoint["tokenizer_itos"])

    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    optimizer_state = checkpoint["optimizer_state"]
    learning_rate = float(optimizer_state["param_groups"][0]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)

    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if device.type == "mps" and "torch_mps" in rng_state:
        torch.mps.set_rng_state(rng_state["torch_mps"])

    return model, optimizer, config, tokenizer, int(checkpoint["training_steps"])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    text = args.data.read_text(encoding="utf-8")
    if args.resume is not None:
        model, optimizer, config, tokenizer, completed_steps = load_training_checkpoint(
            args.resume, device
        )
    else:
        tokenizer = CharTokenizer.from_text(text)
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
        completed_steps = 0

    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = int(0.9 * len(encoded))
    splits = {"train": encoded[:split_at], "val": encoded[split_at:]}

    if len(splits["val"]) <= config.block_size + 1:
        raise ValueError("验证集太短，请减小 block_size 或增加文本")

    print(f"device       : {device}")
    print(f"characters   : {len(text):,}")
    print(f"vocab size   : {tokenizer.vocab_size}")
    print(f"parameters   : {model.count_parameters():,}")
    print(f"context      : {config.block_size} characters")
    if args.resume is not None:
        print(f"resumed from : {args.resume} at step {completed_steps}")

    started = time.perf_counter()
    model.train()
    for additional_step in range(args.steps + 1):
        step = completed_steps + additional_step
        if additional_step % args.eval_interval == 0 or additional_step == args.steps:
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
        if additional_step == args.steps:
            break

        inputs, targets = make_batch(
            splits["train"], args.batch_size, config.block_size, device
        )
        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total_steps = completed_steps + args.steps
    save_training_checkpoint(
        args.output,
        model,
        optimizer,
        config,
        tokenizer,
        total_steps,
        device,
    )
    elapsed = time.perf_counter() - started
    print(f"saved        : {args.output}")
    print(f"elapsed      : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
