"""Tests for the dense-matcher interface (``PairMatches`` + ``match_pair``).

These pin the contract that distinguishes the two wrapper kinds:
- sparse extractors implement ``_extract`` and return a ``MethodOutput``;
- dense matchers implement ``match_pair`` and return a ``PairMatches``.

``MethodWrapper`` provides a default for each that raises ``NotImplementedError``,
so a wrapper that forgets to implement its kind fails loudly rather than silently.
"""

import pytest
import torch

from wrappers.wrapper import MethodWrapper, MethodOutput, PairMatches


class _FakeSparse(MethodWrapper):
    """Minimal sparse extractor: returns two fixed keypoints."""

    def __init__(self):
        """Initialize the fake sparse extractor on CPU without AMP."""
        super().__init__(name="fake-sparse", device="cpu", use_amp=False)

    def _extract(self, img, max_kpts, custom_kpts=None):
        """Return a MethodOutput with two fixed keypoints and random descriptors."""
        kpts = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        return MethodOutput(kpts=kpts, des=torch.randn(2, 4))


class _FakeDense(MethodWrapper):
    """Minimal dense matcher: returns three fixed correspondences.

    Crucially it does *not* implement ``_extract`` -- it must still be
    instantiable, which proves ``_extract`` is no longer abstract.
    """

    def __init__(self):
        """Initialize the fake dense matcher and clear the sparse flag."""
        super().__init__(name="fake-dense", device="cpu", use_amp=False)
        self.is_sparse_feature_extractor = False

    def match_pair(self, img1_path, img2_path, max_kpts=4096):
        """Return a PairMatches with three fixed correspondences."""
        idx = torch.arange(3).repeat(2, 1).T
        return PairMatches(idx, torch.zeros(3, 2), torch.ones(3, 2) * 5.0)


def test_pair_matches_is_namedtuple_and_unpacks():
    """PairMatches supports attribute access and 3-tuple unpacking."""
    pm = PairMatches(torch.zeros(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2)
    # attribute access
    assert torch.equal(pm.kpts2, torch.ones(2, 2) * 2)
    # backward-compatible 3-tuple unpacking
    matches, kpts1, kpts2 = pm
    assert torch.equal(matches, torch.zeros(2, 2))
    assert torch.equal(kpts1, torch.ones(2, 2))
    assert len(pm) == 3


def test_sparse_wrapper_defaults_to_sparse_flag():
    """A sparse wrapper defaults is_sparse_feature_extractor to True."""
    w = _FakeSparse()
    assert w.is_sparse_feature_extractor is True


def test_sparse_extract_returns_method_output():
    """A sparse wrapper's extract returns a MethodOutput with the right shape."""
    w = _FakeSparse()
    img = torch.zeros(3, 32, 32)
    out = w.extract(img, max_kpts=10)
    assert isinstance(out, MethodOutput)
    assert out.kpts.shape == (2, 2)


def test_dense_wrapper_instantiable_without_extract():
    """A dense wrapper is instantiable without implementing _extract."""
    # _extract must not be abstract anymore, or this construction raises TypeError.
    w = _FakeDense()
    assert w.is_sparse_feature_extractor is False


def test_dense_match_pair_returns_pair_matches_and_unpacks():
    """A dense wrapper's match_pair returns an unpackable PairMatches."""
    w = _FakeDense()
    result = w.match_pair("a.png", "b.png", max_kpts=100)
    assert isinstance(result, PairMatches)
    matches, kpts1, kpts2 = result  # call sites rely on tuple unpacking
    assert matches.shape == (3, 2)
    assert torch.equal(kpts2, torch.ones(3, 2) * 5.0)


def test_base_match_pair_raises_not_implemented():
    """The base match_pair raises NotImplementedError for a sparse wrapper."""
    # A sparse wrapper that never implemented match_pair must fail loudly.
    w = _FakeSparse()
    with pytest.raises(NotImplementedError):
        w.match_pair("a.png", "b.png", max_kpts=100)


def test_base_extract_raises_not_implemented_for_dense():
    """The base _extract raises NotImplementedError for a dense wrapper."""
    # A dense wrapper has no sparse _extract; calling it must fail loudly.
    w = _FakeDense()
    with pytest.raises(NotImplementedError):
        w._extract(torch.zeros(3, 16, 16), max_kpts=10)
