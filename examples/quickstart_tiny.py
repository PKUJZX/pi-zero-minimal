import torch

from mini_pi0 import MiniPiZero, SyntheticDataset, tiny_config


def main() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    config.num_inference_steps = 2

    dataset = SyntheticDataset(
        config=config,
        num_samples=4,
        seed=0,
        text_token_ids=(2, 3),
    )
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}

    model = MiniPiZero(config).eval()
    with torch.no_grad():
        loss = model.compute_loss(
            batch["input_ids"],
            batch["pixel_values"],
            batch["attention_mask"],
            batch["proprio"],
            batch["action"],
            eps=torch.zeros_like(batch["action"]),
            t=torch.zeros(1),
        )
        sampled_action = model.sample_actions_cached(
            batch["input_ids"],
            batch["pixel_values"],
            batch["attention_mask"],
            batch["proprio"],
            noise=torch.zeros_like(batch["action"]),
        )

    print(f"tiny_loss={loss.item():.6f}")
    print(f"sampled_action_shape={tuple(sampled_action.shape)}")
    print(f"sampled_action_mean={sampled_action.mean().item():.6f}")


if __name__ == "__main__":
    main()
