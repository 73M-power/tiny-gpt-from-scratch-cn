"""Tiny GPT 的全部模型组件。

实现刻意保持直接：字符分词、手写因果自注意力、Transformer Block，
以及 next-token prediction 所需的线性输出层。
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CharTokenizer:
    """最小字符级 tokenizer：每个不同字符对应一个整数。"""

    UNK = "<unk>"

    def __init__(self, chars: List[str]):
        vocab = sorted(set(chars))
        if self.UNK not in vocab:
            vocab.append(self.UNK)
        self.itos = vocab
        self.stoi = {token: index for index, token in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @classmethod
    def from_itos(cls, itos: List[str]) -> "CharTokenizer":
        # checkpoint 中的编号顺序必须原样恢复，不能重新排序。
        tokenizer = cls.__new__(cls)
        tokenizer.itos = list(itos)
        tokenizer.stoi = {
            token: index for index, token in enumerate(tokenizer.itos)
        }
        if cls.UNK not in tokenizer.stoi:
            raise ValueError("checkpoint 词表缺少 <unk>")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        unknown = self.stoi[self.UNK]
        return [self.stoi.get(char, unknown) for char in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(
            "□" if self.itos[index] == self.UNK else self.itos[index]
            for index in ids
        )


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    因果遮罩保证位置 t 只能看到位置 <= t 的 token，不能偷看未来答案。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))
        self.last_attention: Optional[Tensor] = None

    def forward(self, x: Tensor, capture_attention: bool = False) -> Tensor:
        batch, time, channels = x.shape

        # (B, T, C) -> (B, heads, T, head_size)
        q = self.query(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.causal_mask[:, :, :time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        if capture_attention:
            self.last_attention = weights.detach().cpu()
        weights = self.attn_dropout(weights)

        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, capture_attention: bool = False) -> Tensor:
        x = x + self.attention(self.ln_1(x), capture_attention=capture_attention)
        x = x + self.feed_forward(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 输入 embedding 与输出投影共享参数，是现代语言模型常见做法。
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        token_ids: Tensor,
        targets: Optional[Tensor] = None,
        capture_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        _, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError(
                f"序列长度 {time} 超过上下文窗口 {self.config.block_size}"
            )

        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, capture_attention=capture_attention)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            # 每个位置预测下一个字符；交叉熵内部包含 softmax。
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

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
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
