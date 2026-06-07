"""Characterization test for MethodWrapper.normalize_image.

These golden values lock its behavior across the layout/grayscale refactor. We use
a minimal concrete subclass (no model) so we can call it as a normal bound method.
"""

import torch

from wrappers.wrapper import MethodOutput, MethodWrapper


class _BareWrapper(MethodWrapper):
    """Minimal concrete wrapper that skips model setup, for pure-method tests."""

    def __init__(self):
        pass

    def _extract(self, img, max_kpts, custom_kpts=None) -> MethodOutput:
        raise NotImplementedError


def normalize_image(_self_ignored, *args, **kwargs):
    return _BareWrapper().normalize_image(*args, **kwargs)


def _fixed_inputs():
    torch.manual_seed(0)
    return {
        "chw": torch.rand(3, 4, 5),
        "hwc": torch.rand(4, 5, 3),
        "nchw": torch.rand(2, 3, 4, 5),
    }


def test_normalize_image_golden():
    x = _fixed_inputs()
    gray = normalize_image(None, x["chw"])
    assert tuple(gray.shape) == (1, 4, 5)
    assert round(gray.sum().item(), 5) == 9.42243

    ms = normalize_image(None, x["chw"], mean=0.5, std=0.25)
    assert tuple(ms.shape) == (3, 4, 5)
    assert round(ms.sum().item(), 5) == -8.74185

    ms_per = normalize_image(None, x["nchw"], mean=[0.1, 0.2, 0.3], std=[0.4, 0.5, 0.6])
    assert tuple(ms_per.shape) == (2, 3, 4, 5)
    assert round(ms_per.sum().item(), 5) == 69.46012

    gray_hwc = normalize_image(None, x["hwc"])
    assert tuple(gray_hwc.shape) == (1, 5, 3)
    assert round(gray_hwc.sum().item(), 5) == 7.63114
