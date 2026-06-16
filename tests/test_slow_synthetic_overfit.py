import os
from pathlib import Path

import pytest

from mini_pi0.train import build_arg_parser, run_training


@pytest.mark.slow
def test_manual_tiny_synthetic_overfit_smoke(tmp_path: Path) -> None:
    if os.environ.get("RUN_MINI_PI0_SLOW") != "1":
        pytest.skip("set RUN_MINI_PI0_SLOW=1 to run slow local acceptance checks")

    args = build_arg_parser().parse_args(
        [
            "--model-size",
            "tiny",
            "--device",
            "cpu",
            "--num-steps",
            "25",
            "--save-every",
            "25",
            "--log-every",
            "25",
            "--batch-size",
            "4",
            "--num-samples",
            "16",
            "--lr",
            "0.001",
            "--output-dir",
            str(tmp_path),
            "--dtype",
            "float32",
        ]
    )
    losses = run_training(args)
    assert len(losses) == 25
    assert (tmp_path / "step_000025.pt").exists()
