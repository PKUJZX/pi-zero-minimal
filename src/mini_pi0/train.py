import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import PiZeroConfig
from .data import SyntheticDataset
from .paligemma import load_paligemma
from .pizero import MiniPiZero


def tiny_config() -> PiZeroConfig:
    return PiZeroConfig(
        vision_hidden_size=8,
        vision_intermediate_size=16,
        vision_num_hidden_layers=1,
        vision_num_attention_heads=2,
        image_size=8,
        patch_size=4,
        num_image_tokens=4,
        projection_dim=16,
        vlm_hidden_size=16,
        vlm_intermediate_size=32,
        vlm_num_hidden_layers=2,
        vlm_num_attention_heads=2,
        vlm_num_key_value_heads=1,
        vlm_head_dim=4,
        action_hidden_size=12,
        action_intermediate_size=24,
        action_num_hidden_layers=2,
        action_num_attention_heads=2,
        action_num_key_value_heads=1,
        action_head_dim=4,
        max_text_len=3,
        max_seq_len=7,
        horizon=4,
        action_dim=5,
        proprio_dim=6,
        time_hidden_size=8,
        vocab_size=32,
        image_token_index=31,
        pad_token_id=0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bare synthetic-data training loop for MiniPiZero."
    )
    parser.add_argument("--model-size", choices=("full", "tiny"), default="full")
    parser.add_argument(
        "--paligemma-checkpoint",
        default="checkpoints/paligemma-3b-pt-224",
    )
    parser.add_argument("--output-dir", default="logs/mini-pi0/synthetic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--freeze", choices=("none", "vision", "vlm"), default="none")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.model_size == "full" and device.type == "cpu":
        raise RuntimeError(
            "full model training requires CUDA; use --device cuda or "
            "--model-size tiny for a CPU smoke run"
        )
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")


def make_autocast_context(device: torch.device, dtype: str):
    if device.type == "cuda" and dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def apply_freeze(model: MiniPiZero, freeze: str) -> None:
    if freeze == "none":
        return
    if freeze == "vision":
        modules = [model.paligemma.vision_tower]
    elif freeze == "vlm":
        modules = [model.paligemma]
    else:
        raise ValueError(f"unknown freeze mode: {freeze}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = False


def build_model(args: argparse.Namespace, device: torch.device) -> MiniPiZero:
    config = PiZeroConfig() if args.model_size == "full" else tiny_config()
    model = MiniPiZero(config)
    if args.model_size == "full":
        load_paligemma(model.paligemma, args.paligemma_checkpoint)
    apply_freeze(model, args.freeze)
    return model.to(device)


def build_dataset(args: argparse.Namespace, config: PiZeroConfig) -> SyntheticDataset:
    text_token_ids = (2, 3) if args.model_size == "tiny" else None
    return SyntheticDataset(
        config=config,
        num_samples=args.num_samples,
        seed=args.seed,
        text_token_ids=text_token_ids,
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True) for key, value in batch.items()
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: MiniPiZero,
    optimizer: torch.optim.Optimizer,
    config: PiZeroConfig,
    args: argparse.Namespace,
    loss: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "args": vars(args).copy(),
            "loss": loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_training(args: argparse.Namespace) -> list[float]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    validate_args(args, device)

    model = build_model(args, device)
    dataset = build_dataset(args, model.config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    data_iter = iter(dataloader)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable parameters remain after applying --freeze")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    losses: list[float] = []
    model.train()
    for step in range(1, args.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with make_autocast_context(device, args.dtype):
            loss = model.compute_loss(
                batch["input_ids"],
                batch["pixel_values"],
                batch["attention_mask"],
                batch["proprio"],
                batch["action"],
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.num_steps:
            print(
                f"step={step:06d} loss={loss_value:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
        if step % args.save_every == 0 or step == args.num_steps:
            checkpoint_path = save_checkpoint(
                output_dir,
                step,
                model,
                optimizer,
                model.config,
                args,
                loss_value,
            )
            print(f"saved_checkpoint={checkpoint_path}")

    return losses


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
