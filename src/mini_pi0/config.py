from dataclasses import dataclass

# Shape contract for the minimal PiZero implementation:
# image:    (B, 3, 224, 224), uint8 or float image normalized to [-1, 1]
# img_emb:  (B, 256, 2048)
# text:     (B, <=20), token ids; prompt = "<image>"*256 + <bos> + instruction + "\n"
# proprio: (B, 1, proprio_dim) -> (B, 1, 1024)
# action:   (B, 4, action_dim) -> (B, 4, 1024)
# velocity: (B, 4, action_dim)


@dataclass
class PiZeroConfig:
    # SigLIP vision tower.
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    vision_num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    vision_layer_norm_eps: float = 1e-6
    vision_attention_dropout: float = 0.0
    num_image_tokens: int = 256

    # PaliGemma projector and Gemma VLM.
    projection_dim: int = 2048
    vlm_hidden_size: int = 2048
    vlm_intermediate_size: int = 16384
    vlm_num_hidden_layers: int = 18
    vlm_num_attention_heads: int = 8
    vlm_num_key_value_heads: int = 1
    vlm_head_dim: int = 256
    vlm_rope_theta: float = 10000.0

    # Action expert.
    action_hidden_size: int = 1024
    action_intermediate_size: int = 4096
    action_num_hidden_layers: int = 18
    action_num_attention_heads: int = 8
    action_num_key_value_heads: int = 1
    action_head_dim: int = 256
    rope_theta_action: float = 10000.0

    # Shared Gemma-style block settings.
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # VLA sequence and flow-matching settings.
    max_text_len: int = 20
    max_seq_len: int = 276
    horizon: int = 4
    action_dim: int = 7
    proprio_dim: int = 8
    num_inference_steps: int = 10
    time_hidden_size: int = 256
    time_max_period: float = 10000.0
    final_action_clip_value: float = 1.0

    # Token IDs from google/paligemma-3b-pt-224.
    vocab_size: int = 257216
    image_token_index: int = 257152
    pad_token_id: int = 0
