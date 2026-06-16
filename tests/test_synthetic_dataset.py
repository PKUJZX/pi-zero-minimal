import torch
from torch.utils.data import DataLoader

from mini_pi0 import SyntheticDataset

from .helpers import tiny_config


def test_synthetic_dataset_contract_and_determinism() -> None:
    config = tiny_config()
    dataset = SyntheticDataset(
        config=config,
        num_samples=8,
        seed=123,
        text_token_ids=(2, 3),
    )
    same_dataset = SyntheticDataset(
        config=config,
        num_samples=8,
        seed=123,
        text_token_ids=(2, 3),
    )

    sample = dataset[0]
    seq_len = config.num_image_tokens + config.max_text_len
    assert sample["input_ids"].shape == (seq_len,)
    assert sample["attention_mask"].shape == (seq_len,)
    assert sample["pixel_values"].shape == (
        config.vision_num_channels,
        config.image_size,
        config.image_size,
    )
    assert sample["proprio"].shape == (1, config.proprio_dim)
    assert sample["action"].shape == (config.horizon, config.action_dim)
    assert sample["pixel_values"].min() >= -1.0
    assert sample["pixel_values"].max() <= 1.0
    assert sample["proprio"].min() >= -1.0
    assert sample["proprio"].max() <= 1.0
    assert sample["action"].min() >= -1.0
    assert sample["action"].max() <= 1.0
    assert torch.equal(sample["input_ids"], same_dataset[0]["input_ids"])
    assert torch.allclose(sample["action"], same_dataset[0]["action"])


def test_synthetic_dataset_dataloader_shapes() -> None:
    config = tiny_config()
    dataset = SyntheticDataset(
        config=config,
        num_samples=8,
        seed=0,
        text_token_ids=(2, 3),
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))

    assert batch["input_ids"].shape == (
        4,
        config.num_image_tokens + config.max_text_len,
    )
    assert batch["pixel_values"].shape == (
        4,
        config.vision_num_channels,
        config.image_size,
        config.image_size,
    )
    assert batch["proprio"].shape == (4, 1, config.proprio_dim)
    assert batch["action"].shape == (4, config.horizon, config.action_dim)
