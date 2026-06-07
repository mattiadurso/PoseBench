"""Tests for the mutual-nearest-neighbor matcher in matchers.mnn."""

import torch

from matchers.mnn import MNN, Matches
from matchers.mnn import mutual_nearest_neighbors_from_score_matrix as _mnn_from_score


def test_ratio_test_branch_golden():
    # Pins the ratio_test branch of mutual_nearest_neighbors_from_score_matrix
    # on a fixed-seed score matrix: stricter ratio_test admits fewer matches.
    torch.manual_seed(0)
    d0 = torch.nn.functional.normalize(torch.rand(6, 8), dim=1)
    d1 = torch.nn.functional.normalize(torch.rand(7, 8), dim=1)
    score = (d0 @ d1.T)[None]
    counts = {
        rt: int(_mnn_from_score(score, min_score=-1.0, ratio_test=rt).sum())
        for rt in (1.0, 0.95, 0.8)
    }
    assert counts == {1.0: 3, 0.95: 2, 0.8: 0}


def test_identical_descriptors_match_one_to_one():
    # Orthonormal descriptors -> each row's best match is its own column.
    des = torch.eye(4)
    matcher = MNN(min_score=-1.0)
    (result,) = matcher.match([des], [des])
    assert isinstance(result, Matches)
    matches = result.matches
    assert matches.shape == (4, 2)
    # Every keypoint matches the one with the same index.
    assert torch.equal(matches[:, 0].sort().values, torch.arange(4))
    assert torch.equal(matches[:, 0], matches[:, 1])


def test_matching_matrix_property_reflects_matches():
    des = torch.eye(3)
    matcher = MNN(min_score=-1.0)
    (result,) = matcher.match([des], [des])
    mm = result.matching_matrix
    assert mm.dtype == torch.bool
    assert torch.equal(mm, torch.eye(3, dtype=torch.bool))


def test_min_score_filters_weak_matches():
    # Descriptors with low inner product should be filtered by a high min_score.
    des0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    des1 = torch.tensor([[0.6, 0.0], [0.0, 0.6]])  # inner products are 0.6
    high = MNN(min_score=0.9).match([des0], [des1])[0]
    low = MNN(min_score=0.1).match([des0], [des1])[0]
    assert high.matches.shape[0] == 0
    assert low.matches.shape[0] == 2


def test_empty_descriptors_yield_no_matches():
    des0 = torch.zeros(0, 4)
    des1 = torch.eye(4)
    (result,) = MNN(min_score=-1.0).match([des0], [des1])
    assert result.matches.shape == (0, 2)
    assert result.score_matrix.shape == (0, 4)
