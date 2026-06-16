import torch

from mini_pi0 import MiniPiZero

from .helpers import make_tiny_inputs, tiny_config


def test_tiny_model_forward_loss_and_sampling() -> None:
    torch.manual_seed(0)
    config = tiny_config(num_inference_steps=2)
    model = MiniPiZero(config).eval()
    batch = make_tiny_inputs(config)
    t = torch.tensor([0.0, 0.5])

    velocity = model.forward_velocity(
        batch["input_ids"],
        batch["pixel_values"],
        batch["attention_mask"],
        batch["proprio"],
        batch["action"],
        t,
    )
    assert velocity.shape == (2, config.horizon, config.action_dim)
    assert torch.isfinite(velocity).all()

    loss = model.compute_loss(
        batch["input_ids"],
        batch["pixel_values"],
        batch["attention_mask"],
        batch["proprio"],
        batch["action"],
        t=t,
        eps=torch.zeros_like(batch["action"]),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    sampled = model.sample_actions(
        batch["input_ids"],
        batch["pixel_values"],
        batch["attention_mask"],
        batch["proprio"],
        noise=torch.zeros_like(batch["action"]),
    )
    assert sampled.shape == (2, config.horizon, config.action_dim)
    assert torch.isfinite(sampled).all()


def test_tiny_cached_sampling_matches_no_cache_sampling() -> None:
    torch.manual_seed(0)
    config = tiny_config(num_inference_steps=3)
    model = MiniPiZero(config).eval()
    batch = make_tiny_inputs(config)
    noise = torch.randn(2, config.horizon, config.action_dim)

    no_cache = model.sample_actions(
        batch["input_ids"],
        batch["pixel_values"],
        batch["attention_mask"],
        batch["proprio"],
        noise=noise,
    )
    cached = model.sample_actions_cached(
        batch["input_ids"],
        batch["pixel_values"],
        batch["attention_mask"],
        batch["proprio"],
        noise=noise,
    )

    assert torch.allclose(cached, no_cache, atol=1e-5, rtol=1e-5)
