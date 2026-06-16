import math

import torch
from torch.utils.data import Dataset

from .config import PiZeroConfig


class SyntheticDataset(Dataset):
    """Fixed-size synthetic dataset for overfitting the MiniPiZero training path."""

    default_text_token_ids = (2, 3, 4, 5, 6)

    def __init__(
        self,
        config: PiZeroConfig | None = None,
        num_samples: int = 200,
        seed: int = 0,
        text_token_ids: tuple[int, ...] | None = None,
    ):
        self.config = config if config is not None else PiZeroConfig()
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        token_ids = (
            self.default_text_token_ids if text_token_ids is None else text_token_ids
        )
        self.text_token_ids = tuple(int(token_id) for token_id in token_ids)
        self._validate_text_token_ids()

        generator = torch.Generator()
        generator.manual_seed(seed)

        image_shape = (
            self.num_samples,
            self.config.vision_num_channels,
            self.config.image_size,
            self.config.image_size,
        )
        self.pixel_values = (
            torch.rand(
                image_shape,
                generator=generator,
                dtype=torch.float32,
            )
            .mul(2.0)
            .sub(1.0)
        )
        self.proprio = (
            torch.rand(
                self.num_samples,
                1,
                self.config.proprio_dim,
                generator=generator,
                dtype=torch.float32,
            )
            .mul(2.0)
            .sub(1.0)
        )

        action_out_dim = self.config.horizon * self.config.action_dim
        self.action_weight = torch.randn(
            action_out_dim,
            self.config.proprio_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(self.config.proprio_dim)
        self.action_bias = 0.1 * torch.randn(
            action_out_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.action = self.actions_from_proprio(self.proprio)

        input_ids, attention_mask = self._build_prompt_tensors()
        self.input_ids = input_ids.unsqueeze(0).expand(self.num_samples, -1).clone()
        self.attention_mask = (
            attention_mask.unsqueeze(0).expand(self.num_samples, -1).clone()
        )

    def _validate_text_token_ids(self) -> None:
        if len(self.text_token_ids) == 0:
            raise ValueError("text_token_ids must not be empty")
        if len(self.text_token_ids) > self.config.max_text_len:
            raise ValueError(
                f"text_token_ids length must be <= max_text_len={self.config.max_text_len}"
            )
        for token_id in self.text_token_ids:
            if token_id < 0 or token_id >= self.config.vocab_size:
                raise ValueError(f"text token id {token_id} is outside the vocab")
            if token_id == self.config.pad_token_id:
                raise ValueError("text_token_ids must not include pad_token_id")
            if token_id == self.config.image_token_index:
                raise ValueError("text_token_ids must not include image_token_index")

    def _build_prompt_tensors(self) -> tuple[torch.LongTensor, torch.LongTensor]:
        seq_len = self.config.num_image_tokens + self.config.max_text_len
        input_ids = torch.full(
            (seq_len,),
            self.config.pad_token_id,
            dtype=torch.long,
        )
        input_ids[: self.config.num_image_tokens] = self.config.image_token_index

        text_start = self.config.num_image_tokens
        text_end = text_start + len(self.text_token_ids)
        input_ids[text_start:text_end] = torch.tensor(
            self.text_token_ids,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(seq_len, dtype=torch.long)
        attention_mask[:text_end] = 1
        return input_ids, attention_mask

    def actions_from_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 3:
            raise ValueError(
                f"proprio must have shape (B, 1, proprio_dim), got {tuple(proprio.shape)}"
            )
        if proprio.shape[1:] != (1, self.config.proprio_dim):
            raise ValueError(
                "proprio must have shape "
                f"(B, 1, {self.config.proprio_dim}), got {tuple(proprio.shape)}"
            )

        weight = self.action_weight.to(device=proprio.device, dtype=proprio.dtype)
        bias = self.action_bias.to(device=proprio.device, dtype=proprio.dtype)
        action = torch.tanh(proprio[:, 0, :] @ weight.T + bias)
        return action.view(
            proprio.shape[0], self.config.horizon, self.config.action_dim
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "pixel_values": self.pixel_values[index],
            "proprio": self.proprio[index],
            "action": self.action[index],
        }
