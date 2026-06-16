import torch
import torch.nn as nn
import torch.nn.functional as F

from .paligemma import PaliGemmaForConditionalGeneration, merge_image_text_embeddings
from .vla_modules import (
    ActionDecoder,
    ActionEncoder,
    ActionExpert,
    ProprioEncoder,
    SinusoidalTimeEmbedding,
    build_causal_mask_and_position_ids,
    joint_forward,
    joint_forward_action_with_cache,
    prefill_vlm_proprio_cache,
)


def make_flow_matching_sample(
    action: torch.FloatTensor,
    t: torch.FloatTensor | None = None,
    eps: torch.FloatTensor | None = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if action.ndim != 3:
        raise ValueError(
            f"action must have shape (B, horizon, action_dim), got {tuple(action.shape)}"
        )
    if not action.is_floating_point():
        raise ValueError(f"action must be floating point, got {action.dtype}")

    batch_size = action.shape[0]
    if t is None:
        t = torch.rand(batch_size, device=action.device, dtype=action.dtype)
    else:
        t = t.to(device=action.device, dtype=action.dtype)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.shape != (batch_size,):
        raise ValueError(f"t must have shape ({batch_size},) or ({batch_size}, 1)")

    if eps is None:
        eps = torch.randn_like(action)
    else:
        eps = eps.to(device=action.device, dtype=action.dtype)
    if eps.shape != action.shape:
        raise ValueError(
            f"eps must have shape {tuple(action.shape)}, got {tuple(eps.shape)}"
        )

    t_broadcast = t[:, None, None]
    noisy_action = (1.0 - t_broadcast) * eps + t_broadcast * action
    target_velocity = action - eps
    return noisy_action, target_velocity, t, eps


class MiniPiZero(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config)
        self.time_embedding = SinusoidalTimeEmbedding(
            config.time_hidden_size,
            max_period=config.time_max_period,
        )
        self.action_encoder = ActionEncoder(config)
        self.proprio_encoder = ProprioEncoder(config)
        self.action_expert = ActionExpert(config)
        self.action_decoder = ActionDecoder(config)

    def _attention_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask must have shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        return attention_mask.to(device=input_ids.device)

    def embed_image_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        text_embeds = self.paligemma.get_input_embeddings()(input_ids)
        pixel_values = pixel_values.to(
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        vision_hidden_states = self.paligemma.vision_tower(pixel_values)
        image_features = self.paligemma.multi_modal_projector(vision_hidden_states)
        return merge_image_text_embeddings(
            input_ids,
            text_embeds,
            image_features,
            image_token_index=self.config.image_token_index,
            num_image_tokens=self.config.num_image_tokens,
        )

    def forward_velocity(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        proprio: torch.FloatTensor,
        action: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        attention_mask = self._attention_mask(input_ids, attention_mask)
        vlm_embeds = self.embed_image_text(input_ids, pixel_values)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            build_causal_mask_and_position_ids(
                attention_mask,
                self.config,
                vlm_embeds.dtype,
            )
        )

        t = t.to(device=action.device, dtype=action.dtype)
        time_embeds = self.time_embedding(t)
        proprio_embeds = self.proprio_encoder(proprio)
        action_embeds = self.action_encoder(action, time_embeds)
        action_hidden_states = joint_forward(
            self.paligemma.language_model.model.layers,
            self.action_expert,
            vlm_embeds,
            proprio_embeds,
            action_embeds,
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
        )
        return self.action_decoder(action_hidden_states)

    def compute_loss(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        proprio: torch.FloatTensor,
        action: torch.FloatTensor,
        t: torch.FloatTensor | None = None,
        eps: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        noisy_action, target_velocity, t, _ = make_flow_matching_sample(action, t, eps)
        pred_velocity = self.forward_velocity(
            input_ids,
            pixel_values,
            attention_mask,
            proprio,
            noisy_action,
            t,
        )
        return F.mse_loss(pred_velocity, target_velocity)

    def _initial_action_sample(
        self,
        input_ids: torch.LongTensor,
        proprio: torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
    ) -> tuple[int, torch.FloatTensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (B, image_text_len), got {tuple(input_ids.shape)}"
            )

        batch_size = input_ids.shape[0]
        action_shape = (batch_size, self.config.horizon, self.config.action_dim)
        if noise is None:
            action = torch.randn(
                action_shape,
                device=proprio.device,
                dtype=proprio.dtype,
            )
        else:
            action = noise.to(device=proprio.device, dtype=proprio.dtype)
            if action.shape != action_shape:
                raise ValueError(
                    f"noise must have shape {action_shape}, got {tuple(action.shape)}"
                )
        return batch_size, action

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        proprio: torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        batch_size, action = self._initial_action_sample(input_ids, proprio, noise)
        num_steps = self.config.num_inference_steps
        if num_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx * dt,
                device=action.device,
                dtype=action.dtype,
            )
            velocity = self.forward_velocity(
                input_ids,
                pixel_values,
                attention_mask,
                proprio,
                action,
                t,
            )
            action = action + dt * velocity

        if self.config.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.config.final_action_clip_value,
                self.config.final_action_clip_value,
            )
        return action

    @torch.no_grad()
    def sample_actions_cached(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        proprio: torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        batch_size, action = self._initial_action_sample(input_ids, proprio, noise)
        attention_mask = self._attention_mask(input_ids, attention_mask)
        vlm_embeds = self.embed_image_text(input_ids, pixel_values)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            build_causal_mask_and_position_ids(
                attention_mask,
                self.config,
                vlm_embeds.dtype,
            )
        )
        proprio_embeds = self.proprio_encoder(proprio)
        prefix_cache = prefill_vlm_proprio_cache(
            self.paligemma.language_model.model.layers,
            self.action_expert,
            vlm_embeds,
            proprio_embeds,
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
        )

        action_start = input_ids.shape[1] + proprio_embeds.shape[1]
        action_attention_mask = causal_mask[:, :, action_start:, :]
        num_steps = self.config.num_inference_steps
        if num_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx * dt,
                device=action.device,
                dtype=action.dtype,
            )
            time_embeds = self.time_embedding(t)
            action_embeds = self.action_encoder(action, time_embeds)
            action_hidden_states = joint_forward_action_with_cache(
                self.action_expert,
                action_embeds,
                prefix_cache,
                action_attention_mask,
                action_position_ids,
            )
            velocity = self.action_decoder(action_hidden_states)
            action = action + dt * velocity

        if self.config.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.config.final_action_clip_value,
                self.config.final_action_clip_value,
            )
        return action
