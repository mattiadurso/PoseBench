"""Characterization test for the hpatches results-dict builder.

GPU benchmark runs verify the full pipeline, but this CPU test pins the extracted
``_hpatches_threshold_metrics`` logic (the per-threshold dict assembly) so the
refactor is checked even when no GPU is available.
"""

import math

import pandas as pd

from benchmarks_2D.hpatches.hpatches_benchmark import _hpatches_threshold_metrics


def _row():
    return pd.Series(
        {
            "mean_repeatability": 0.5,
            "median_repeatability": 0.4,
            "mean_matching_score": 0.3,
            "median_matching_score": 0.25,
            "mean_matching_accuracy": 0.6,
            "median_matching_accuracy": 0.55,
            "mean_n_matches_proposed": 100.0,
            "mean_n_inliers_nn_GT": 60.0,
            "mean_precision": 0.6,
            "mean_recall": 0.7,
        }
    )


def _homography_data():
    return pd.DataFrame(
        {
            "accuracy_thr": [3.0, 3.0, 5.0],
            "ransac_thr": [0.5, 1.0, 0.5],
            "homography_accuracy": [0.8, 0.9, 0.95],
        }
    )


def test_threshold_metrics_picks_best_ransac():
    m = _hpatches_threshold_metrics(_row(), _homography_data(), 3)
    assert m["repeatability_3"] == {"mean": 0.5, "median": 0.4}
    assert m["matching_score_3"] == {"mean": 0.3, "median": 0.25}
    assert m["matching_accuracy_3"] == {"mean": 0.6, "median": 0.55}
    # accuracy_thr==3 has two rows; best homography_accuracy is 0.9 at ransac_thr 1.0
    assert m["homography_accuracy_3"] == {
        "mean": 0.9,
        "median": 0.9,
        "best_ransac_thr": 1.0,
    }
    assert m["match_stats_3"] == {
        "total_matches_mean": 100.0,
        "inliers_mean": 60.0,
        "precision_mean": 0.6,
        "recall_mean": 0.7,
    }


def test_threshold_metrics_no_homography_rows():
    m = _hpatches_threshold_metrics(_row(), _homography_data(), 7)  # no accuracy_thr==7
    assert m["homography_accuracy_7"] == {
        "mean": 0.0,
        "median": 0.0,
        "best_ransac_thr": None,
    }


def test_threshold_metrics_nan_repeatability():
    row = _row()
    row["mean_repeatability"] = float("nan")
    m = _hpatches_threshold_metrics(row, _homography_data(), 3)
    assert math.isnan(m["repeatability_3"]["mean"])
