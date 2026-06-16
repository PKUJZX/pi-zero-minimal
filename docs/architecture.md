# Architecture Notes

`mini-pi0` keeps the model path intentionally small: one image, one short text
prompt, one proprio token, and a short action chunk.

## Tensor Contract

| Name | Shape | Meaning |
|---|---|---|
| `pixel_values` | `(B, 3, 224, 224)` | Image normalized to `[-1, 1]` |
| image tokens | `(B, 256, 2048)` | SigLIP patches projected into Gemma width |
| `input_ids` | `(B, 256 + max_text_len)` | Image placeholder tokens plus right-padded text |
| `proprio` | `(B, 1, proprio_dim)` | Robot state token |
| `action` | `(B, horizon, action_dim)` | Noisy action chunk during training or sampling |
| velocity | `(B, horizon, action_dim)` | Flow matching velocity prediction |

The default teaching setup uses `horizon=4`, `action_dim=7`, and
`proprio_dim=8`. The tiny config used by tests shrinks every width and image size
while keeping the same interfaces.

## Blocks And Visibility

The sequence is split into three blocks:

```text
A = image + text tokens
B = proprio token
C = action tokens
```

The additive attention mask implements:

| Query block | Can attend to A | Can attend to B | Can attend to C |
|---|---:|---:|---:|
| A: VLM | yes | no | no |
| B: proprio | yes | yes | no |
| C: action | yes | yes | yes |

Text padding columns are blocked for proprio and action as well. Tests assert the
block values directly because mask bugs usually do not crash; they silently
change model behavior.

## Flow Matching

Training samples a timestep `t` in `[0, 1]`, Gaussian noise `eps`, and interpolates
from noise to action:

```text
x_t = (1 - t) * eps + t * action
target_velocity = action - eps
loss = mse(model(x_t, t), target_velocity)
```

Sampling starts from Gaussian noise and integrates from `t=0` to `t=1` with Euler
updates:

```text
x <- x + dt * velocity(x, t)
```

## KV Cache

The no-cache sampler recomputes VLM, proprio, and action tokens at every
denoising step. The cached sampler pre-fills the VLM and proprio K/V tensors once
and only recomputes action-token Q/K/V at each step. In float32, the tiny tests
require cached and uncached sampling to match within a tight tolerance.
