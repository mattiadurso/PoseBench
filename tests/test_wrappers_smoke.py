"""Smoke tests one layer above the pure helpers.

These verify that wrappers are discoverable, the registry maps names to classes,
and the reference wrapper (disk-kornia) runs end-to-end on a synthetic image and
produces a valid ``MethodOutput`` that the MNN matcher can consume. Wrappers that
need a GPU or vendored weights are skipped when unavailable, so the suite stays
runnable anywhere.
"""

import pytest
import torch

from matchers.mnn import MNN, Matches
from wrappers.wrapper import MethodOutput
from wrappers_manager import get_wrappers_list, wrappers_manager


def test_get_wrappers_list_discovers_known_methods():
    wrappers = get_wrappers_list()
    assert isinstance(wrappers, list)
    assert "disk" in wrappers  # the reference method
    # The base and example wrappers must not be exposed as methods.
    assert "wrapper" not in wrappers
    assert "example" not in wrappers


def test_unknown_wrapper_raises():
    with pytest.raises(ValueError):
        wrappers_manager("definitely-not-a-method")


def _synthetic_image(device, size=256, seed=0):
    """float32 CxHxW image in [0, 1], dims multiple of 16 (as load_image yields)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, size, size, generator=g).to(device)


@pytest.fixture(scope="module")
def disk_wrapper():
    if not torch.cuda.is_available():
        pytest.skip("disk-kornia requires CUDA")
    try:
        return wrappers_manager("disk-kornia", device="cuda")
    except Exception as exc:  # missing weights / no network
        pytest.skip(f"disk-kornia unavailable: {exc}")


def test_disk_kornia_extract_end_to_end(disk_wrapper):
    img = _synthetic_image("cuda")
    out = disk_wrapper.extract(img, max_kpts=256)
    assert isinstance(out, MethodOutput)
    assert out.kpts.ndim == 2 and out.kpts.shape[1] == 2
    assert out.des is not None and out.des.shape[0] == out.kpts.shape[0]
    # Keypoints lie inside the (border-trimmed) image.
    H, W = img.shape[-2:]
    assert bool((out.kpts[:, 0] >= 0).all()) and bool((out.kpts[:, 0] <= W).all())
    assert bool((out.kpts[:, 1] >= 0).all()) and bool((out.kpts[:, 1] <= H).all())


def test_disk_kornia_extract_and_match(disk_wrapper):
    out0 = disk_wrapper.extract(_synthetic_image("cuda", seed=0), max_kpts=256)
    out1 = disk_wrapper.extract(_synthetic_image("cuda", seed=1), max_kpts=256)
    (result,) = MNN(min_score=-1.0).match([out0.des.cpu()], [out1.des.cpu()])
    assert isinstance(result, Matches)
    assert result.matches.ndim == 2 and result.matches.shape[1] == 2
