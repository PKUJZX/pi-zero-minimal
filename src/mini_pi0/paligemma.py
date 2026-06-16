import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from .gemma import GemmaDecoderLayer, GemmaRMSNorm
from .siglip import SiglipVisionModel


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_hidden_size,
            config.projection_dim,
            bias=True,
        )
        self.output_scale = config.vlm_hidden_size**-0.5

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.linear(image_features)
        image_features = image_features * self.output_scale
        return image_features


class GemmaTokenEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.vlm_hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.embed_scale = math.sqrt(config.vlm_hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale


def merge_image_text_embeddings(
    input_ids: torch.LongTensor,
    text_embeds: torch.Tensor,
    image_features: torch.Tensor,
    *,
    image_token_index: int,
    num_image_tokens: int,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")
    if text_embeds.ndim != 3:
        raise ValueError(
            f"text_embeds must be 3D, got shape {tuple(text_embeds.shape)}"
        )
    if image_features.ndim != 3:
        raise ValueError(
            f"image_features must be 3D, got shape {tuple(image_features.shape)}"
        )

    batch_size, seq_len = input_ids.shape
    if seq_len < num_image_tokens:
        raise ValueError(
            f"input sequence length {seq_len} is shorter than {num_image_tokens} image tokens"
        )

    expected_text_shape = (batch_size, seq_len)
    if text_embeds.shape[:2] != expected_text_shape:
        raise ValueError(
            f"text_embeds must start with {expected_text_shape}, got {tuple(text_embeds.shape)}"
        )

    expected_image_shape = (batch_size, num_image_tokens, text_embeds.shape[-1])
    if image_features.shape != expected_image_shape:
        raise ValueError(
            f"image_features must be {expected_image_shape}, got {tuple(image_features.shape)}"
        )

    image_token_prefix = input_ids[:, :num_image_tokens]
    if not torch.all(image_token_prefix == image_token_index):
        raise ValueError("the first num_image_tokens input ids must be image tokens")
    if torch.any(input_ids[:, num_image_tokens:] == image_token_index):
        raise ValueError("image tokens are only supported in the image prefix")

    merged = text_embeds.clone()
    merged[:, :num_image_tokens, :] = image_features.to(
        device=text_embeds.device,
        dtype=text_embeds.dtype,
    )
    return merged


class PaliGemmaInputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = GemmaTokenEmbeddings(config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def get_image_features(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.multi_modal_projector(vision_hidden_states)

    def forward(
        self,
        input_ids: torch.LongTensor,
        vision_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        text_embeds = self.embed_tokens(input_ids)
        image_features = self.get_image_features(vision_hidden_states)
        return merge_image_text_embeddings(
            input_ids,
            text_embeds,
            image_features,
            image_token_index=self.config.image_token_index,
            num_image_tokens=self.config.num_image_tokens,
        )


class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.vlm_hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx=layer_idx, prefix="vlm")
                for layer_idx in range(config.vlm_num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.vlm_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
        if position_ids.size(0) == 1 and batch_size != 1:
            position_ids = position_ids.expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.zeros(
                batch_size,
                1,
                seq_len,
                seq_len,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        normalizer = torch.tensor(
            self.config.vlm_hidden_size**0.5,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        hidden_states = inputs_embeds * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        return self.norm(hidden_states)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.vlm_hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        text_embeds = self.get_input_embeddings()(input_ids)
        vision_hidden_states = self.vision_tower(pixel_values.type_as(text_embeds))
        image_features = self.multi_modal_projector(vision_hidden_states)
        inputs_embeds = merge_image_text_embeddings(
            input_ids,
            text_embeds,
            image_features,
            image_token_index=self.config.image_token_index,
            num_image_tokens=self.config.num_image_tokens,
        )

        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                1,
                seq_len + 1,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            if input_ids.shape[0] != 1:
                position_ids = position_ids.expand(input_ids.shape[0], -1)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        filenames = sorted(set(index["weight_map"].values()))
        return [checkpoint_dir / filename for filename in filenames]
    return sorted(checkpoint_dir.glob("model-*.safetensors"))


def load_paligemma(model: PaliGemmaForConditionalGeneration, ckpt_dir: str | Path):
    checkpoint_dir = Path(ckpt_dir)
    state_dict = {}
    for checkpoint_file in _checkpoint_files(checkpoint_dir):
        with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    embed_key = "language_model.model.embed_tokens.weight"
    lm_head_key = "language_model.lm_head.weight"
    if lm_head_key not in state_dict:
        state_dict[lm_head_key] = state_dict[embed_key]

    load_result = model.load_state_dict(state_dict, strict=True)
    model.tie_weights()
    return load_result
