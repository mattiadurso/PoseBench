"""Characterization tests locking the MNN matching-statistics functions.

These pin the exact current output of ``Matches.compute_scores_stats`` and
``compute_correct_wrong_mismatched_inexistent_unsure_matches`` on a fixed
synthetic input, so the helper-extraction refactor is provably behavior-preserving.
"""

import math

import torch

from matchers.mnn import (
    Matches,
    compute_correct_wrong_mismatched_inexistent_unsure_matches as ccw,
)

GOLDEN = {
    "n_matches_GT": 4,
    "n_matches_proposed": 3,
    "n_matches_correct": 2,
    "n_matches_wrong": 1,
    "n_matches_mismatched": 1,
    "n_matches_inexistent": 0,
    "n_matches_unsure": 0,
    "mean_GT_score": 0.573957,
    "mean_proposed_score": 0.561523,
    "mean_correct_score": 0.525245,
    "mean_wrong_score": 0.634079,
    "mean_mismatched_score": 0.634079,
    "mean_inexistent_score": float("nan"),
    "mean_unsure_score": float("nan"),
    "mean_matching_matrix_score": 0.429807,
    "matches_precision": 0.666667,
    "matches_recall": 0.5,
    "mean_margin_proposed": 0.145088,
    "mean_margin_correct": 0.086449,
    "mean_margin_wrong": 0.262366,
    "mean_margin_mismatched": 0.262366,
    "mean_margin_inexistent": float("nan"),
    "mean_ratio_proposed": 0.829186,
    "mean_ratio_correct": 0.890116,
    "mean_ratio_wrong": 0.707326,
    "mean_ratio_mismatched": 0.707326,
    "mean_ratio_inexistent": float("nan"),
    "n_masked": -1,
}


def _fixture():
    torch.manual_seed(0)
    score = torch.rand(4, 5)
    matches = torch.tensor([[0, 1], [1, 0], [3, 4]])
    gt = torch.zeros(5, 6, dtype=torch.bool)
    gt[0, 1] = gt[1, 2] = gt[2, 0] = gt[3, 4] = True
    gt[2, 5] = True
    return Matches(matches, score), gt


def test_compute_scores_stats_matches_golden():
    m, gt = _fixture()
    stats = m.compute_scores_stats(gt)
    assert set(stats.keys()) == set(GOLDEN.keys())
    for key, expected in GOLDEN.items():
        got = stats[key]
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(got), f"{key} expected NaN, got {got}"
        else:
            assert math.isclose(got, expected, rel_tol=1e-4, abs_tol=1e-4), (
                f"{key}: got {got}, expected {expected}"
            )


def test_ccw_category_counts_match_golden():
    m, gt = _fixture()
    out = ccw(m.matching_matrix[None], gt[None])[0]
    counts = {
        a: int(getattr(out, a).sum())
        for a in ["proposed", "correct", "wrong", "mismatched", "inexistent", "unsure"]
    }
    assert counts == {
        "proposed": 3,
        "correct": 2,
        "wrong": 1,
        "mismatched": 1,
        "inexistent": 0,
        "unsure": 0,
    }
