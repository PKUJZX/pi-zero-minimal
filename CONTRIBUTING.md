# Contributing

Thanks for improving `mini-pi0`. This project is intentionally small, so changes
should preserve readability and verifiability.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before Opening A Pull Request

Run the fast checks:

```bash
ruff format --check .
ruff check .
python -m pytest tests -m "not slow and not checkpoint"
```

If your change touches checkpoint loading, also run:

```bash
MINI_PI0_PALIGEMMA_CKPT=checkpoints/paligemma-3b-pt-224 \
  python -m pytest tests -m checkpoint
```

## Checkpoint Policy

Do not commit model weights, generated checkpoints, logs, videos, or datasets.
Use `checkpoints/` for local weights; it is ignored by git.

## Scope

Good first-release changes include tests, examples, docs, bug fixes, and clearer
shape/mask logic. Large training stack features such as RLDS/TFDS, DDP, LoRA,
quantization, simulator adapters, and real-data pipelines should be discussed in
an issue first.
