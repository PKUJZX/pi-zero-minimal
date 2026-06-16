from pathlib import Path

import torch

from mini_pi0.train import build_arg_parser, run_training


def test_tiny_train_loop_writes_checkpoint(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--model-size",
            "tiny",
            "--device",
            "cpu",
            "--num-steps",
            "2",
            "--save-every",
            "2",
            "--log-every",
            "1",
            "--batch-size",
            "2",
            "--num-samples",
            "8",
            "--output-dir",
            str(tmp_path),
            "--dtype",
            "float32",
        ]
    )
    losses = run_training(args)

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    checkpoint_path = tmp_path / "step_000002.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "args" in checkpoint
    assert torch.isfinite(torch.tensor(checkpoint["loss"]))
