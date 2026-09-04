"""不依赖 pytest 的最小单元测试：python -m unittest -v。"""

from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch

import train
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

    def test_training_checkpoint_restores_model_optimizer_and_metadata(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        tokens = torch.tensor([self.tokenizer.encode("语言模型")])
        _, loss = self.model(tokens, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        save_checkpoint = getattr(train, "save_training_checkpoint", None)
        load_checkpoint = getattr(train, "load_training_checkpoint", None)
        self.assertIsNotNone(
            save_checkpoint,
            "train.py 需要提供 save_training_checkpoint",
        )
        self.assertIsNotNone(
            load_checkpoint,
            "train.py 需要提供 load_training_checkpoint",
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "resume.pt"
            save_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=7,
                device=torch.device("cpu"),
            )
            (
                restored_model,
                restored_optimizer,
                restored_config,
                restored_tokenizer,
                restored_steps,
            ) = load_checkpoint(checkpoint_path, torch.device("cpu"))

        self.assertEqual(restored_config, self.config)
        self.assertEqual(restored_tokenizer.itos, self.tokenizer.itos)
        self.assertEqual(restored_steps, 7)
        for name, expected in self.model.state_dict().items():
            self.assertTrue(torch.equal(restored_model.state_dict()[name], expected))

        original_optimizer_state = next(iter(optimizer.state.values()))
        restored_optimizer_state = next(iter(restored_optimizer.state.values()))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(
                torch.equal(restored_optimizer_state[key], original_optimizer_state[key])
            )

    def test_training_checkpoint_restores_random_states(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        random.seed(101)
        torch.manual_seed(202)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "resume.pt"
            train.save_training_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=3,
                device=torch.device("cpu"),
            )
            expected_python_random = random.random()
            expected_torch_random = torch.rand(4)

            train.load_training_checkpoint(checkpoint_path, torch.device("cpu"))

        self.assertEqual(random.random(), expected_python_random)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch_random))

    def test_resume_rejects_inference_only_checkpoint_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "inference-only.pt"
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "config": self.config.to_dict(),
                    "tokenizer_itos": self.tokenizer.itos,
                    "training_steps": 7,
                },
                checkpoint_path,
            )

            try:
                train.load_training_checkpoint(checkpoint_path, torch.device("cpu"))
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("optimizer_state", str(error))
                self.assertIn("rng_state", str(error))
            else:
                self.fail("仅含推理状态的 checkpoint 不应被当作完整训练状态恢复")

    def test_train_cli_resumes_and_accumulates_training_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            data_path = temporary / "tiny.txt"
            full_checkpoint = temporary / "full.pt"
            first_checkpoint = temporary / "first.pt"
            resumed_checkpoint = temporary / "resumed.pt"
            data_path.write_text("语言模型学习。\n" * 20, encoding="utf-8")

            runtime_arguments = [
                sys.executable,
                "train.py",
                "--data",
                str(data_path),
                "--batch-size",
                "2",
                "--eval-interval",
                "1",
                "--eval-iters",
                "1",
                "--device",
                "cpu",
            ]
            model_arguments = [
                "--block-size",
                "4",
                "--n-layer",
                "1",
                "--n-head",
                "1",
                "--n-embd",
                "8",
                "--dropout",
                "0.2",
                "--learning-rate",
                "0.001",
            ]
            full_run = subprocess.run(
                runtime_arguments
                + model_arguments
                + ["--steps", "3", "--output", str(full_checkpoint)],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(full_run.returncode, 0, full_run.stderr)

            first_run = subprocess.run(
                runtime_arguments
                + model_arguments
                + ["--steps", "1", "--output", str(first_checkpoint)],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(first_run.returncode, 0, first_run.stderr)

            resumed_run = subprocess.run(
                runtime_arguments
                + [
                    "--resume",
                    str(first_checkpoint),
                    "--block-size",
                    "64",
                    "--n-layer",
                    "2",
                    "--n-head",
                    "2",
                    "--n-embd",
                    "16",
                    "--dropout",
                    "0.9",
                    "--learning-rate",
                    "0.5",
                    "--steps",
                    "2",
                    "--output",
                    str(resumed_checkpoint),
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resumed_run.returncode, 0, resumed_run.stderr)

            full_state = torch.load(
                full_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                resumed_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(full_state["training_steps"], 3)
            self.assertEqual(resumed_state["training_steps"], 3)
            self.assertEqual(resumed_state["config"]["n_layer"], 1)
            self.assertEqual(resumed_state["config"]["n_embd"], 8)
            self.assertEqual(
                resumed_state["optimizer_state"]["param_groups"][0]["lr"],
                0.001,
            )

            self.assertEqual(
                full_state["model_state"].keys(),
                resumed_state["model_state"].keys(),
            )
            for name, expected in full_state["model_state"].items():
                self.assertTrue(
                    torch.equal(resumed_state["model_state"][name], expected),
                    name,
                )

            full_optimizer = full_state["optimizer_state"]
            resumed_optimizer = resumed_state["optimizer_state"]
            self.assertEqual(
                resumed_optimizer["param_groups"],
                full_optimizer["param_groups"],
            )
            self.assertEqual(resumed_optimizer["state"].keys(), full_optimizer["state"].keys())
            for parameter_id, expected_state in full_optimizer["state"].items():
                actual_state = resumed_optimizer["state"][parameter_id]
                self.assertEqual(actual_state.keys(), expected_state.keys())
                for name, expected in expected_state.items():
                    actual = actual_state[name]
                    if isinstance(expected, torch.Tensor):
                        self.assertTrue(torch.equal(actual, expected), name)
                    else:
                        self.assertEqual(actual, expected)

            self.assertEqual(
                resumed_state["rng_state"]["python"],
                full_state["rng_state"]["python"],
            )
            self.assertTrue(
                torch.equal(
                    resumed_state["rng_state"]["torch_cpu"],
                    full_state["rng_state"]["torch_cpu"],
                )
            )

    def test_evaluation_does_not_advance_training_random_state(self) -> None:
        data = torch.tensor(self.tokenizer.encode("语言模型学习。" * 8))
        splits = {"train": data, "val": data}
        torch.manual_seed(404)
        state_before_evaluation = torch.get_rng_state().clone()

        train.estimate_loss(
            self.model,
            splits,
            batch_size=2,
            block_size=4,
            eval_iters=2,
            device=torch.device("cpu"),
        )

        self.assertTrue(
            torch.equal(torch.get_rng_state(), state_before_evaluation),
            "评估抽样不应改变后续训练所使用的全局随机状态",
        )

    def test_train_cli_rejects_resume_source_as_output(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            data_path = temporary / "tiny.txt"
            checkpoint_path = temporary / "resume.pt"
            data_path.write_text("语言模型学习。\n" * 30, encoding="utf-8")
            train.save_training_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=3,
                device=torch.device("cpu"),
            )
            checkpoint_before_command = checkpoint_path.read_bytes()

            result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--data",
                    str(data_path),
                    "--resume",
                    str(checkpoint_path),
                    "--output",
                    str(checkpoint_path),
                    "--steps",
                    "0",
                    "--batch-size",
                    "2",
                    "--eval-iters",
                    "1",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--output", result.stderr)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before_command)

    def test_train_cli_rejects_negative_additional_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "invalid.pt"
            result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--steps",
                    "-1",
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--steps", result.stderr)
            self.assertFalse(output_path.exists())

    def test_resume_rejects_incomplete_random_state_with_clear_error(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "incomplete-rng.pt"
            train.save_training_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=3,
                device=torch.device("cpu"),
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            del checkpoint["rng_state"]["torch_cpu"]
            torch.save(checkpoint, checkpoint_path)

            try:
                train.load_training_checkpoint(checkpoint_path, torch.device("cpu"))
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("torch_cpu", str(error))
            else:
                self.fail("缺少 Torch RNG 的 checkpoint 不应进入恢复流程")

    def test_resume_requires_the_same_training_device_type(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cpu.pt"
            train.save_training_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=3,
                device=torch.device("cpu"),
            )

            try:
                train.load_training_checkpoint(checkpoint_path, torch.device("mps"))
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("cpu", str(error))
                self.assertIn("mps", str(error))
            else:
                self.fail("严格 resume 不应静默切换训练设备类型")

    def test_resume_rejects_missing_mps_random_state(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "incomplete-mps.pt"
            train.save_training_checkpoint(
                checkpoint_path,
                self.model,
                optimizer,
                self.config,
                self.tokenizer,
                training_steps=3,
                device=torch.device("cpu"),
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["rng_state"]["device_type"] = "mps"
            torch.save(checkpoint, checkpoint_path)

            try:
                train.load_training_checkpoint(checkpoint_path, torch.device("mps"))
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("torch_mps", str(error))
            else:
                self.fail("缺少 MPS RNG 的 checkpoint 不应被严格恢复")


if __name__ == "__main__":
    unittest.main()
