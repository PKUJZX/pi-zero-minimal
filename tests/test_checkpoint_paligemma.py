import os
from pathlib import Path

import pytest

from mini_pi0.config import PiZeroConfig
from mini_pi0.paligemma import PaliGemmaForConditionalGeneration, load_paligemma


@pytest.mark.checkpoint
def test_paligemma_checkpoint_loads_from_env() -> None:
    checkpoint_dir = os.environ.get("MINI_PI0_PALIGEMMA_CKPT")
    if not checkpoint_dir:
        pytest.skip("set MINI_PI0_PALIGEMMA_CKPT to run checkpoint tests")

    checkpoint_path = Path(checkpoint_dir)
    assert checkpoint_path.exists(), checkpoint_path

    model = PaliGemmaForConditionalGeneration(PiZeroConfig()).eval()
    load_result = load_paligemma(model, checkpoint_path)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
