import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_gemma_dims(config, prefix: str) -> dict[str, int | float | bool]:
    if prefix == "vlm":
        return {
            "hidden_size": config.vlm_hidden_size,
            "intermediate_size": config.vlm_intermediate_size,
            "num_attention_heads": config.vlm_num_attention_heads,
            "num_key_value_heads": config.vlm_num_key_value_heads,
            "head_dim": config.vlm_head_dim,
            "rope_theta": config.vlm_rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": config.attention_bias,
            "attention_dropout": config.attention_dropout,
        }
    if prefix == "action":
        return {
            "hidden_size": config.action_hidden_size,
            "intermediate_size": config.action_intermediate_size,
            "num_attention_heads": config.action_num_attention_heads,
            "num_key_value_heads": config.action_num_key_value_heads,
            "head_dim": config.action_head_dim,
            "rope_theta": config.rope_theta_action,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": config.attention_bias,
            "attention_dropout": config.attention_dropout,
        }
    raise ValueError(f"Unsupported Gemma config prefix: {prefix}")


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        seq_len,
        head_dim,
    )


class GemmaMLP(nn.Module):
    def __init__(self, config, prefix: str = "vlm"):
        super().__init__()
        dims = _get_gemma_dims(config, prefix)
        self.hidden_size = int(dims["hidden_size"])
        self.intermediate_size = int(dims["intermediate_size"])

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class GemmaAttention(nn.Module):
    def __init__(self, config, layer_idx: int = 0, prefix: str = "vlm"):
        super().__init__()
        dims = _get_gemma_dims(config, prefix)
        self.layer_idx = layer_idx
        self.attention_dropout = float(dims["attention_dropout"])
        self.hidden_size = int(dims["hidden_size"])
        self.num_heads = int(dims["num_attention_heads"])
        self.head_dim = int(dims["head_dim"])
        self.num_key_value_heads = int(dims["num_key_value_heads"])
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        attention_bias = bool(dims["attention_bias"])
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            base=float(dims["rope_theta"]),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            raise ValueError("position_ids must be provided")

        batch_size, query_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, query_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(
                query_states,
                key_states.transpose(2, 3),
            )
            * self.scaling
        )

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (
            batch_size,
            self.num_heads,
            query_len,
            self.head_dim,
        ):
            raise ValueError(
                f"attn_output should be {(batch_size, self.num_heads, query_len, self.head_dim)}, "
                f"got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, -1)
        return self.o_proj(attn_output)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int = 0, prefix: str = "vlm"):
        super().__init__()
        dims = _get_gemma_dims(config, prefix)
        hidden_size = int(dims["hidden_size"])
        rms_norm_eps = float(dims["rms_norm_eps"])

        self.hidden_size = hidden_size
        self.self_attn = GemmaAttention(config, layer_idx=layer_idx, prefix=prefix)
        self.mlp = GemmaMLP(config, prefix=prefix)
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
