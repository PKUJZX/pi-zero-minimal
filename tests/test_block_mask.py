import torch

from mini_pi0.vla_modules import build_causal_mask_and_position_ids

from .helpers import tiny_config


def assert_all_value(tensor: torch.Tensor, value: float) -> None:
    assert torch.all(tensor == value)


def test_block_wise_mask_visibility_and_padding() -> None:
    config = tiny_config(horizon=3)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    causal_mask, vlm_pos, proprio_pos, action_pos = build_causal_mask_and_position_ids(
        attention_mask,
        config,
        torch.float32,
    )

    mask = causal_mask[0, 0]
    blocked = torch.finfo(torch.float32).min
    image_text_len = attention_mask.shape[1]
    valid_vlm_end = 3
    proprio_start = image_text_len
    proprio_end = proprio_start + 1
    action_start = proprio_end

    assert_all_value(mask[:valid_vlm_end, :valid_vlm_end], 0.0)
    assert_all_value(mask[:valid_vlm_end, valid_vlm_end:], blocked)

    assert_all_value(mask[proprio_start:proprio_end, :valid_vlm_end], 0.0)
    assert_all_value(
        mask[proprio_start:proprio_end, valid_vlm_end:proprio_start], blocked
    )
    assert_all_value(mask[proprio_start:proprio_end, proprio_start:proprio_end], 0.0)
    assert_all_value(mask[proprio_start:proprio_end, action_start:], blocked)

    assert_all_value(mask[action_start:, :valid_vlm_end], 0.0)
    assert_all_value(mask[action_start:, valid_vlm_end:proprio_start], blocked)
    assert_all_value(mask[action_start:, proprio_start:], 0.0)

    assert vlm_pos.shape == attention_mask.shape
    assert torch.equal(proprio_pos, torch.ones(1, 1, dtype=torch.long))
    assert torch.equal(action_pos, torch.tensor([[2, 3, 4]], dtype=torch.long))


def test_block_wise_mask_requires_right_padding() -> None:
    config = tiny_config()
    attention_mask = torch.tensor([[1, 0, 1, 0, 0, 0, 0]], dtype=torch.long)

    try:
        build_causal_mask_and_position_ids(attention_mask, config, torch.float32)
    except ValueError as exc:
        assert "right padding" in str(exc)
    else:
        raise AssertionError("expected right-padding validation to fail")
