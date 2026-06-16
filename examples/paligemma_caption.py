import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from mini_pi0 import PiZeroConfig, load_paligemma
from mini_pi0.paligemma import PaliGemmaForConditionalGeneration


def make_prefix_lm_mask(
    seq_len: int,
    prefix_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    visible = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    visible[:prefix_len, :prefix_len] = True
    for row in range(prefix_len, seq_len):
        visible[row, : row + 1] = True

    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask = mask.masked_fill(visible, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption an image with mini PaliGemma."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/paligemma-3b-pt-224",
        help="Local google/paligemma-3b-pt-224 directory.",
    )
    parser.add_argument(
        "--image",
        default="media/maniskill_pp.png",
        help="Image path.",
    )
    parser.add_argument("--prompt", default="<image>caption en")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    device = resolve_device(args.device)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(images=image, text=args.prompt, return_tensors="pt")
    prefix_len = model_inputs["input_ids"].shape[1]
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    model = PaliGemmaForConditionalGeneration(PiZeroConfig()).to(device).eval()
    load_result = load_paligemma(model, checkpoint)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(load_result)

    input_ids = model_inputs["input_ids"]
    pixel_values = model_inputs["pixel_values"]
    eos_token_id = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            seq_len = input_ids.shape[1]
            attention_mask = make_prefix_lm_mask(
                seq_len,
                prefix_len,
                device=device,
                dtype=torch.float32,
            )
            position_ids = torch.arange(
                1,
                seq_len + 1,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            logits = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if next_token.item() == eos_token_id:
                break

    new_tokens = input_ids[:, prefix_len:]
    caption = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(caption)


if __name__ == "__main__":
    main()
