from __future__ import annotations

import torch

from mini_pi0.config import PiZeroConfig
from mini_pi0.train import tiny_config as _tiny_config


def tiny_config(**overrides) -> PiZeroConfig:
    values = _tiny_config().__dict__.copy()
    values.update(overrides)
    return PiZeroConfig(**values)


def make_tiny_inputs(
    config: PiZeroConfig | None = None,
    batch_size: int = 2,
) -> dict[str, torch.Tensor]:
    config = config or tiny_config()
    generator = torch.Generator()
    generator.manual_seed(0)

    seq_len = config.num_image_tokens + config.max_text_len
    input_ids = torch.full((batch_size, seq_len), config.pad_token_id, dtype=torch.long)
    input_ids[:, : config.num_image_tokens] = config.image_token_index
    input_ids[:, config.num_image_tokens : config.num_image_tokens + 2] = torch.tensor(
        [2, 3],
        dtype=torch.long,
    )

    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, : config.num_image_tokens + 2] = 1

    pixel_values = (
        torch.rand(
            batch_size,
            config.vision_num_channels,
            config.image_size,
            config.image_size,
            generator=generator,
        )
        .mul(2.0)
        .sub(1.0)
    )
    proprio = (
        torch.rand(
            batch_size,
            1,
            config.proprio_dim,
            generator=generator,
        )
        .mul(2.0)
        .sub(1.0)
    )
    action = (
        torch.rand(
            batch_size,
            config.horizon,
            config.action_dim,
            generator=generator,
        )
        .mul(2.0)
        .sub(1.0)
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "proprio": proprio,
        "action": action,
    }
