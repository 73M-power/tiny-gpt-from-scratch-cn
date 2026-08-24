"""加载 checkpoint，让训练后的 Tiny GPT 续写文本。"""

import argparse
from pathlib import Path

import torch

from tinygpt import CharTokenizer, GPTConfig, TinyGPT
from train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Tiny GPT 生成文本")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinygpt.pt"))
    parser.add_argument("--prompt", default="模型学习")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer.from_itos(checkpoint["tokenizer_itos"])
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(
        prompt_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
