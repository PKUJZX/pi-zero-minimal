import pytest
import torch

from mini_pi0.pizero import make_flow_matching_sample


def test_make_flow_matching_sample_uses_pi0_direction() -> None:
    action = torch.tensor([[[1.0, -1.0], [0.5, 0.25]]])
    eps = torch.tensor([[[-1.0, 1.0], [0.25, -0.75]]])
    t = torch.tensor([0.25])

    noisy_action, target_velocity, returned_t, returned_eps = make_flow_matching_sample(
        action,
        t=t,
        eps=eps,
    )

    expected_noisy = (1.0 - t[:, None, None]) * eps + t[:, None, None] * action
    assert torch.allclose(noisy_action, expected_noisy)
    assert torch.allclose(target_velocity, action - eps)
    assert torch.equal(returned_t, t)
    assert torch.equal(returned_eps, eps)


def test_make_flow_matching_sample_validates_shapes() -> None:
    action = torch.zeros(2, 4, 7)
    with pytest.raises(ValueError, match="t must have shape"):
        make_flow_matching_sample(action, t=torch.zeros(3))
    with pytest.raises(ValueError, match="eps must have shape"):
        make_flow_matching_sample(action, eps=torch.zeros(2, 4, 8))
    with pytest.raises(ValueError, match="action must have shape"):
        make_flow_matching_sample(torch.zeros(2, 7))
