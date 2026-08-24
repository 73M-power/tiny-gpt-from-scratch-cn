"""不依赖 pytest 的最小单元测试：python -m unittest -v。"""

import unittest

import torch

from tinygpt import CharTokenizer, GPTConfig, TinyGPT


class TinyGPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.tokenizer = CharTokenizer.from_text("语言模型学习。")
        self.config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        self.model = TinyGPT(self.config)

    def test_tokenizer_round_trip(self) -> None:
        text = "语言模型。"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        restored = CharTokenizer.from_itos(self.tokenizer.itos)
        self.assertEqual(restored.encode(text), self.tokenizer.encode(text))

    def test_forward_shape_and_loss(self) -> None:
        tokens = torch.tensor([self.tokenizer.encode("语言模型")])
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (1, 4, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_attention_cannot_see_future(self) -> None:
        tokens = torch.tensor([self.tokenizer.encode("语言模型")])
        self.model(tokens, capture_attention=True)
        attention = self.model.blocks[0].attention.last_attention[0, 0]
        future = torch.triu(attention, diagonal=1)
        self.assertTrue(torch.equal(future, torch.zeros_like(future)))

    def test_generate_adds_requested_tokens(self) -> None:
        tokens = torch.tensor([self.tokenizer.encode("语言")])
        generated = self.model.generate(tokens, max_new_tokens=5, top_k=3)
        self.assertEqual(generated.shape[1], tokens.shape[1] + 5)


if __name__ == "__main__":
    unittest.main()
