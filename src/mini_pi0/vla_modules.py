import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma import (
    GemmaDecoderLayer,
    GemmaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be even")
        if dim < 2:
            raise ValueError("SinusoidalTimeEmbedding dim must be at least 2")

        self.dim = dim
        self.half_dim = dim // 2
        self.max_period = max_period

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,) or (B, 1), got {tuple(t.shape)}")

        exponent = math.log(self.max_period) / (self.half_dim - 1)
        freqs = torch.exp(
            torch.arange(self.half_dim, device=t.device, dtype=t.dtype) * -exponent
        )
        angles = t[:, None] * freqs[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ActionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.hidden_size = config.action_hidden_size
        self.time_hidden_size = config.time_hidden_size

        self.action_proj = nn.Linear(self.action_dim, self.hidden_size)
        self.fuse_proj = nn.Linear(
            self.hidden_size + self.time_hidden_size,
            self.hidden_size,
        )
        self.nonlinearity = nn.SiLU()
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if action.ndim != 3:
            raise ValueError(
                f"action must have shape (B, T, action_dim), got {tuple(action.shape)}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action last dim must be {self.action_dim}, got {action.shape[-1]}"
            )
        if time_emb.ndim != 2:
            raise ValueError(
                f"time_emb must have shape (B, time_hidden_size), got {tuple(time_emb.shape)}"
            )
        if time_emb.shape != (action.shape[0], self.time_hidden_size):
            raise ValueError(
                f"time_emb must be {(action.shape[0], self.time_hidden_size)}, got {tuple(time_emb.shape)}"
            )

        action_emb = self.action_proj(action)
        time_emb = time_emb.unsqueeze(1).expand(-1, action.shape[1], -1)
        hidden_states = torch.cat((time_emb, action_emb), dim=-1)
        hidden_states = self.nonlinearity(self.fuse_proj(hidden_states))
        return self.output_proj(hidden_states)


class ProprioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proprio_dim = config.proprio_dim
        self.hidden_size = config.action_hidden_size
        self.linear = nn.Linear(self.proprio_dim, self.hidden_size)

    def forward(self, proprio: torch.FloatTensor) -> torch.FloatTensor:
        if proprio.ndim != 3:
            raise ValueError(
                f"proprio must have shape (B, T, proprio_dim), got {tuple(proprio.shape)}"
            )
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[-1]}"
            )
        return self.linear(proprio)


class ActionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.action_hidden_size
        self.action_dim = config.action_dim
        self.linear = nn.Linear(self.hidden_size, self.action_dim)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (B, T, hidden_size), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dim must be {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        return self.linear(hidden_states)


def build_causal_mask_and_position_ids(
    attention_mask: torch.Tensor,
    config,
    dtype: torch.dtype,
) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (B, image_text_len), got {tuple(attention_mask.shape)}"
        )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"dtype must be floating point, got {dtype}")

    batch_size, image_text_len = attention_mask.shape
    device = attention_mask.device
    num_proprio_tokens = 1
    num_action_tokens = config.horizon
    proprio_start = image_text_len
    proprio_end = proprio_start + num_proprio_tokens
    action_start = proprio_end
    total_len = image_text_len + num_proprio_tokens + num_action_tokens

    attention_mask_bool = attention_mask.to(torch.bool)
    token_counts = attention_mask_bool.long().sum(dim=1)
    expected_right_padding = torch.arange(image_text_len, device=device).unsqueeze(
        0
    ) < token_counts.unsqueeze(1)
    if not torch.equal(attention_mask_bool, expected_right_padding):
        raise ValueError("attention_mask must use right padding")

    blocked = torch.finfo(dtype).min
    causal_mask = torch.full(
        (batch_size, total_len, total_len),
        blocked,
        dtype=dtype,
        device=device,
    )
    for batch_idx, token_count in enumerate(token_counts.tolist()):
        causal_mask[batch_idx, :token_count, :token_count] = 0.0
        causal_mask[batch_idx, proprio_start:, :token_count] = 0.0

    causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = 0.0
    causal_mask[:, action_start:, proprio_start:] = 0.0
    causal_mask = causal_mask.unsqueeze(1)

    vlm_position_ids = (
        torch.arange(
            1,
            image_text_len + 1,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    proprio_position_ids = torch.ones(
        batch_size,
        num_proprio_tokens,
        device=device,
        dtype=torch.long,
    )
    action_position_ids = (
        torch.arange(
            num_proprio_tokens + 1,
            num_proprio_tokens + num_action_tokens + 1,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids


class ActionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.action_num_hidden_layers != config.vlm_num_hidden_layers:
            raise ValueError("action expert layer count must match the VLM")
        if config.action_num_attention_heads != config.vlm_num_attention_heads:
            raise ValueError("action expert Q head count must match the VLM")
        if config.action_num_key_value_heads != config.vlm_num_key_value_heads:
            raise ValueError("action expert KV head count must match the VLM")
        if config.action_head_dim != config.vlm_head_dim:
            raise ValueError("action expert head_dim must match the VLM")

        self.config = config
        self.hidden_size = config.action_hidden_size
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx=layer_idx, prefix="action")
                for layer_idx in range(config.action_num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.action_hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = math.sqrt(config.action_hidden_size)

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    @property
    def num_heads(self) -> int:
        return self.layers[0].self_attn.num_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds.ndim != 3:
            raise ValueError(
                f"inputs_embeds must have shape (B, T, hidden_size), got {tuple(inputs_embeds.shape)}"
            )
        if inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"inputs_embeds last dim must be {self.hidden_size}, got {inputs_embeds.shape[-1]}"
            )

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

        hidden_states = inputs_embeds * self.embed_scale
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self.norm(hidden_states)


def _expand_position_ids(
    position_ids: torch.LongTensor,
    batch_size: int,
) -> torch.LongTensor:
    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must have shape (B, T), got {tuple(position_ids.shape)}"
        )
    if position_ids.size(0) == 1 and batch_size != 1:
        return position_ids.expand(batch_size, -1)
    if position_ids.size(0) != batch_size:
        raise ValueError(
            f"position_ids batch must be 1 or {batch_size}, got {position_ids.size(0)}"
        )
    return position_ids


def _project_qkv_with_rope(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = hidden_states.shape
    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_ids must be {(batch_size, seq_len)}, got {tuple(position_ids.shape)}"
        )

    query_states = attention.q_proj(hidden_states)
    key_states = attention.k_proj(hidden_states)
    value_states = attention.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size,
        seq_len,
        attention.num_heads,
        attention.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size,
        seq_len,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size,
        seq_len,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)

    cos, sin = attention.rotary_emb(value_states, position_ids)
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
    return query_states, key_states, value_states


def _joint_attention(
    vlm_attention: nn.Module,
    expert_attention: nn.Module,
    vlm_hidden_states: torch.Tensor,
    expert_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    vlm_position_ids: torch.LongTensor,
    expert_position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vlm_attention.num_heads != expert_attention.num_heads:
        raise ValueError("VLM and action expert must use the same number of Q heads")
    if vlm_attention.head_dim != expert_attention.head_dim:
        raise ValueError("VLM and action expert must use the same head_dim")

    batch_size = vlm_hidden_states.shape[0]
    vlm_query_len = vlm_hidden_states.shape[1]
    expert_query_len = expert_hidden_states.shape[1]

    vlm_query, vlm_key, vlm_value = _project_qkv_with_rope(
        vlm_attention,
        vlm_hidden_states,
        vlm_position_ids,
    )
    expert_query, expert_key, expert_value = _project_qkv_with_rope(
        expert_attention,
        expert_hidden_states,
        expert_position_ids,
    )

    vlm_key = repeat_kv(vlm_key, vlm_attention.num_key_value_groups)
    vlm_value = repeat_kv(vlm_value, vlm_attention.num_key_value_groups)
    expert_key = repeat_kv(expert_key, expert_attention.num_key_value_groups)
    expert_value = repeat_kv(expert_value, expert_attention.num_key_value_groups)

    query_states = torch.cat((vlm_query, expert_query), dim=2)
    key_states = torch.cat((vlm_key, expert_key), dim=2)
    value_states = torch.cat((vlm_value, expert_value), dim=2)

    attn_weights = (
        torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        )
        * vlm_attention.scaling
    )
    attention_mask = attention_mask[
        :,
        :,
        : attn_weights.shape[-2],
        : attn_weights.shape[-1],
    ]
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = F.dropout(
        attn_weights,
        p=vlm_attention.attention_dropout,
        training=vlm_attention.training,
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, vlm_query_len + expert_query_len, -1)

    vlm_output, expert_output = torch.split(
        attn_output,
        [vlm_query_len, expert_query_len],
        dim=1,
    )
    return vlm_attention.o_proj(vlm_output), expert_attention.o_proj(expert_output)


def prefill_vlm_proprio_cache(
    vlm_layers: nn.ModuleList,
    action_expert: ActionExpert,
    vlm_embeds: torch.Tensor,
    proprio_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    vlm_position_ids: torch.LongTensor,
    proprio_position_ids: torch.LongTensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(vlm_layers) != len(action_expert.layers):
        raise ValueError("VLM and action expert layer counts must match")
    if vlm_embeds.ndim != 3 or proprio_embeds.ndim != 3:
        raise ValueError("embeddings must have shape (B, T, hidden_size)")

    batch_size = vlm_embeds.shape[0]
    if proprio_embeds.shape[0] != batch_size:
        raise ValueError("all embedding batches must match")

    vlm_position_ids = _expand_position_ids(vlm_position_ids, batch_size)
    proprio_position_ids = _expand_position_ids(proprio_position_ids, batch_size)

    vlm_scale = torch.tensor(
        vlm_embeds.shape[-1] ** 0.5,
        dtype=vlm_embeds.dtype,
        device=vlm_embeds.device,
    )
    expert_scale = torch.tensor(
        action_expert.hidden_size**0.5,
        dtype=proprio_embeds.dtype,
        device=proprio_embeds.device,
    )
    vlm_hidden_states = vlm_embeds * vlm_scale
    proprio_hidden_states = proprio_embeds * expert_scale
    prefix_len = vlm_hidden_states.shape[1] + proprio_hidden_states.shape[1]
    prefix_mask = attention_mask[:, :, :prefix_len, :prefix_len]
    prefix_cache = []

    for vlm_layer, expert_layer in zip(vlm_layers, action_expert.layers):
        vlm_residual = vlm_hidden_states
        proprio_residual = proprio_hidden_states

        vlm_norm = vlm_layer.input_layernorm(vlm_hidden_states)
        proprio_norm = expert_layer.input_layernorm(proprio_hidden_states)

        _, vlm_key, vlm_value = _project_qkv_with_rope(
            vlm_layer.self_attn,
            vlm_norm,
            vlm_position_ids,
        )
        _, proprio_key, proprio_value = _project_qkv_with_rope(
            expert_layer.self_attn,
            proprio_norm,
            proprio_position_ids,
        )
        vlm_key = repeat_kv(vlm_key, vlm_layer.self_attn.num_key_value_groups)
        vlm_value = repeat_kv(vlm_value, vlm_layer.self_attn.num_key_value_groups)
        proprio_key = repeat_kv(
            proprio_key,
            expert_layer.self_attn.num_key_value_groups,
        )
        proprio_value = repeat_kv(
            proprio_value,
            expert_layer.self_attn.num_key_value_groups,
        )
        prefix_cache.append(
            (
                torch.cat((vlm_key, proprio_key), dim=2),
                torch.cat((vlm_value, proprio_value), dim=2),
            )
        )

        vlm_attn, proprio_attn = _joint_attention(
            vlm_layer.self_attn,
            expert_layer.self_attn,
            vlm_norm,
            proprio_norm,
            prefix_mask,
            vlm_position_ids,
            proprio_position_ids,
        )
        vlm_hidden_states = vlm_residual + vlm_attn
        proprio_hidden_states = proprio_residual + proprio_attn

        vlm_residual = vlm_hidden_states
        proprio_residual = proprio_hidden_states
        vlm_hidden_states = vlm_layer.post_attention_layernorm(vlm_hidden_states)
        proprio_hidden_states = expert_layer.post_attention_layernorm(
            proprio_hidden_states
        )
        vlm_hidden_states = vlm_residual + vlm_layer.mlp(vlm_hidden_states)
        proprio_hidden_states = proprio_residual + expert_layer.mlp(
            proprio_hidden_states
        )

    return prefix_cache


def joint_forward_action_with_cache(
    action_expert: ActionExpert,
    action_embeds: torch.Tensor,
    prefix_cache: list[tuple[torch.Tensor, torch.Tensor]],
    action_attention_mask: torch.Tensor,
    action_position_ids: torch.LongTensor,
) -> torch.Tensor:
    if action_embeds.ndim != 3:
        raise ValueError("action_embeds must have shape (B, T, hidden_size)")
    if len(prefix_cache) != len(action_expert.layers):
        raise ValueError("prefix cache length must match action expert layer count")

    batch_size, action_len, _ = action_embeds.shape
    action_position_ids = _expand_position_ids(action_position_ids, batch_size)
    expert_scale = torch.tensor(
        action_expert.hidden_size**0.5,
        dtype=action_embeds.dtype,
        device=action_embeds.device,
    )
    action_hidden_states = action_embeds * expert_scale

    for layer_idx, expert_layer in enumerate(action_expert.layers):
        residual = action_hidden_states
        action_norm = expert_layer.input_layernorm(action_hidden_states)
        query_states, action_key, action_value = _project_qkv_with_rope(
            expert_layer.self_attn,
            action_norm,
            action_position_ids,
        )
        action_key = repeat_kv(
            action_key,
            expert_layer.self_attn.num_key_value_groups,
        )
        action_value = repeat_kv(
            action_value,
            expert_layer.self_attn.num_key_value_groups,
        )

        prefix_key, prefix_value = prefix_cache[layer_idx]
        key_states = torch.cat((prefix_key, action_key), dim=2)
        value_states = torch.cat((prefix_value, action_value), dim=2)

        attn_weights = (
            torch.matmul(
                query_states,
                key_states.transpose(2, 3),
            )
            * expert_layer.self_attn.scaling
        )
        attention_mask = action_attention_mask[
            :,
            :,
            :action_len,
            : attn_weights.shape[-1],
        ]
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights,
            p=expert_layer.self_attn.attention_dropout,
            training=expert_layer.self_attn.training,
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, action_len, -1)
        action_hidden_states = residual + expert_layer.self_attn.o_proj(attn_output)

        residual = action_hidden_states
        action_hidden_states = expert_layer.post_attention_layernorm(
            action_hidden_states
        )
        action_hidden_states = residual + expert_layer.mlp(action_hidden_states)

    return action_expert.norm(action_hidden_states)


def joint_forward(
    vlm_layers: nn.ModuleList,
    action_expert: ActionExpert,
    vlm_embeds: torch.Tensor,
    proprio_embeds: torch.Tensor,
    action_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    vlm_position_ids: torch.LongTensor,
    proprio_position_ids: torch.LongTensor,
    action_position_ids: torch.LongTensor,
) -> torch.Tensor:
    if len(vlm_layers) != len(action_expert.layers):
        raise ValueError("VLM and action expert layer counts must match")
    if vlm_embeds.ndim != 3 or proprio_embeds.ndim != 3 or action_embeds.ndim != 3:
        raise ValueError("embeddings must all have shape (B, T, hidden_size)")

    batch_size = vlm_embeds.shape[0]
    if proprio_embeds.shape[0] != batch_size or action_embeds.shape[0] != batch_size:
        raise ValueError("all embedding batches must match")

    vlm_position_ids = _expand_position_ids(vlm_position_ids, batch_size)
    proprio_position_ids = _expand_position_ids(proprio_position_ids, batch_size)
    action_position_ids = _expand_position_ids(action_position_ids, batch_size)
    expert_position_ids = torch.cat((proprio_position_ids, action_position_ids), dim=1)

    vlm_scale = torch.tensor(
        vlm_embeds.shape[-1] ** 0.5,
        dtype=vlm_embeds.dtype,
        device=vlm_embeds.device,
    )
    expert_scale = torch.tensor(
        action_expert.hidden_size**0.5,
        dtype=action_embeds.dtype,
        device=action_embeds.device,
    )
    vlm_hidden_states = vlm_embeds * vlm_scale
    expert_hidden_states = torch.cat((proprio_embeds, action_embeds), dim=1)
    expert_hidden_states = expert_hidden_states * expert_scale

    for vlm_layer, expert_layer in zip(vlm_layers, action_expert.layers):
        vlm_residual = vlm_hidden_states
        expert_residual = expert_hidden_states

        vlm_norm = vlm_layer.input_layernorm(vlm_hidden_states)
        expert_norm = expert_layer.input_layernorm(expert_hidden_states)
        vlm_attn, expert_attn = _joint_attention(
            vlm_layer.self_attn,
            expert_layer.self_attn,
            vlm_norm,
            expert_norm,
            attention_mask,
            vlm_position_ids,
            expert_position_ids,
        )
        vlm_hidden_states = vlm_residual + vlm_attn
        expert_hidden_states = expert_residual + expert_attn

        vlm_residual = vlm_hidden_states
        expert_residual = expert_hidden_states
        vlm_hidden_states = vlm_layer.post_attention_layernorm(vlm_hidden_states)
        expert_hidden_states = expert_layer.post_attention_layernorm(
            expert_hidden_states
        )
        vlm_hidden_states = vlm_residual + vlm_layer.mlp(vlm_hidden_states)
        expert_hidden_states = expert_residual + expert_layer.mlp(expert_hidden_states)

    expert_hidden_states = action_expert.norm(expert_hidden_states)
    proprio_len = proprio_embeds.shape[1]
    return expert_hidden_states[:, proprio_len:, :]
