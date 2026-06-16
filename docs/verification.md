# Verification

The default test suite is designed for CI and laptops. It does not need GPU,
PaliGemma weights, HuggingFace access, or simulator dependencies.

## Fast Local Suite

```bash
python -m pytest tests -m "not slow and not checkpoint"
ruff format --check .
ruff check .
```

This covers:

- Synthetic dataset shape/range/determinism.
- Flow matching interpolation and target velocity.
- Block-wise attention mask and right-padding validation.
- Tiny end-to-end forward, loss, no-cache sampling, and cached sampling.
- Tiny synthetic training loop and checkpoint writing.

## Checkpoint Suite

Download PaliGemma locally:

```bash
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir checkpoints/paligemma-3b-pt-224
```

If direct HuggingFace access fails:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Run:

```bash
MINI_PI0_PALIGEMMA_CKPT=checkpoints/paligemma-3b-pt-224 \
  python -m pytest tests -m checkpoint
```

## Slow Manual Acceptance

Slow tests are skipped unless explicitly enabled:

```bash
RUN_MINI_PI0_SLOW=1 python -m pytest tests -m slow
```

The historical full synthetic overfit acceptance is recorded in `PROGRESS.md`.
It is intentionally kept out of CI because it requires local hardware and a
large checkpoint.

## Milestone Mapping

| Milestone | Current release check |
|---|---|
| M1: PaliGemma checkpoint loading/text path | `checkpoint` tests and optional caption example |
| M2: block-wise causal mask | `tests/test_block_mask.py` |
| M3: synthetic training path | `tests/test_train_loop.py` plus manual slow acceptance |
| M4: KV cache sampling | `tests/test_tiny_model.py` |
