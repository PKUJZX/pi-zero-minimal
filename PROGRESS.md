# mini-pi0 Progress

Last updated: 2026-06-16, Asia/Shanghai

## Current Status

`mini-pi0` has been converted from a local experiment folder into a standalone
teaching repo with a standard Python package layout, pytest checks, examples,
docs, and CI.

The first release keeps the model semantics unchanged and focuses on packaging,
verification, and discoverability. Real-data loading, official open-pi-zero
checkpoint remapping, SimplerEnv evaluation, DDP, LoRA, quantization, and wandb
remain out of scope for this release.

## Verified Milestones

| Milestone | Status |
|---|---|
| M1 | Mini PaliGemma text generation previously matched HuggingFace token-for-token with the local `google/paligemma-3b-pt-224` checkpoint. |
| M2 | Block-wise VLM/proprio/action mask semantics and padding behavior are covered by pytest. |
| M3 | The historical full-frozen synthetic overfit acceptance reached flow MSE `< 0.01` and sampled-action L1 `< 0.05`. |
| M4 | Cached action sampling matches no-cache sampling in float32 on local checks and tiny pytest coverage. |

## Release Checks

Fast checks:

```bash
ruff format --check .
ruff check .
python -m pytest tests -m "not slow and not checkpoint"
```

Optional checkpoint checks:

```bash
MINI_PI0_PALIGEMMA_CKPT=checkpoints/paligemma-3b-pt-224 \
  python -m pytest tests -m checkpoint
```

Optional slow local checks:

```bash
RUN_MINI_PI0_SLOW=1 python -m pytest tests -m slow
```

## Historical Acceptance Outputs

M1 observed output from the original local workspace:

```text
paligemma_text_generation_device=cuda:0
paligemma_text_generation_prefix_len=260
paligemma_text_generation_hf_ids=[28040, 675, 5744, 575, 573, 5086, 576, 573, 4408, 1]
paligemma_text_generation_mini_ids=[28040, 675, 5744, 575, 573, 5086, 576, 573, 4408, 1]
paligemma_text_generation_hf_text='robot with ball in the center of the target'
paligemma_text_generation_mini_text='robot with ball in the center of the target'
paligemma_text_generation_tokens_match=True
```

M3 observed output from the original local workspace:

```text
full_frozen_trainable_params=313850887
positive_step=002800 train_loss=0.065087 flow_mse=0.009740 sample_l1=0.029949 lr=0.00004658
positive_pass_step=002800 flow_mse=0.009740 sample_l1=0.029949
negative_shuffled_true_rule_l1=0.516809 negative_flow_mse_true_rule=0.868036
synthetic_overfit_checks_passed=True
```

## Local Assets

- Checkpoints belong under `checkpoints/` and are ignored by git.
- Demo image: `media/maniskill_pp.png`.
- Architecture image: `media/open-pi-zero-overview.png`.

If HuggingFace access is unreliable, try:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
