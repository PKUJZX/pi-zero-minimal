from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "vision_hidden_size must be divisible by vision_num_attention_heads"
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.vision_attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be {(batch_size, self.num_heads, seq_len, seq_len)}, "
                f"got {tuple(attn_weights.size())}"
            )

        if attention_mask is not None:
            expected_shape = (batch_size, 1, seq_len, seq_len)
            if attention_mask.size() != expected_shape:
                raise ValueError(
                    f"Attention mask should be {expected_shape}, got {tuple(attention_mask.size())}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be {(batch_size, self.num_heads, seq_len, self.head_dim)}, "
                f"got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            config.vision_hidden_size,
            eps=config.vision_layer_norm_eps,
        )
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            config.vision_hidden_size,
            eps=config.vision_layer_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.vision_num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.vision_hidden_size,
            eps=config.vision_layer_norm_eps,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)
