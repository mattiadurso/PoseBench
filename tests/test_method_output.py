"""Tests for the MethodOutput container in wrappers.wrapper."""

import pytest
import torch

from wrappers.wrapper import MethodOutput


def test_defaults_filled_on_post_init():
    kpts = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = MethodOutput(kpts=kpts)
    assert torch.equal(out.kpts_scores, torch.ones(3))
    assert torch.equal(out.kpts_sizes, torch.ones(3))
    assert torch.equal(out.kpts_scales, torch.ones(3))
    assert torch.equal(out.kpts_angles, torch.zeros(3))


def test_rejects_non_2d_kpts():
    with pytest.raises(AssertionError):
        MethodOutput(kpts=torch.zeros(5))  # 1D, must be (N, 2)


def test_mask_selects_rows():
    kpts = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    des = torch.arange(6).reshape(3, 2).float()
    out = MethodOutput(kpts=kpts, des=des)
    mask = torch.tensor([True, False, True])
    masked = out.mask(mask)
    assert torch.equal(masked.kpts, torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    assert torch.equal(masked.des, torch.tensor([[0.0, 1.0], [4.0, 5.0]]))
    assert masked.kpts_scores.shape == (2,)


def test_mask_requires_bool():
    out = MethodOutput(kpts=torch.zeros(2, 2))
    with pytest.raises(AssertionError):
        out.mask(torch.tensor([1, 0]))  # not bool


def test_getitem_get_and_contains():
    out = MethodOutput(kpts=torch.zeros(2, 2))
    assert "kpts" in out
    assert torch.equal(out["kpts"], torch.zeros(2, 2))
    assert out.get("does_not_exist") is None


def test_cpu_returns_method_output():
    out = MethodOutput(kpts=torch.zeros(2, 2)).cpu()
    assert isinstance(out, MethodOutput)
    assert out.kpts.device.type == "cpu"
