"""Tests for visible, non-silent failure handling in pose estimation.

A failed pair must still produce the 180-degree sentinel result (so the
benchmark keeps going), but the failure must be *visible*: logged at WARNING
level rather than swallowed at debug level or hidden behind a bare ``except``.
"""

import logging

import numpy as np

from benchmarks_2D import utils_benchmark
from benchmarks_2D.utils_benchmark import estimate_pose, process_pose_estimation

FAILURE_RESULT = ("a.png", "b.png", 180, 180, 180, 0)


def _intrinsics():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    return K.copy(), K.copy()


def _pair_data(n_matches):
    """Build a valid ``process_pose_estimation`` input with ``n_matches`` matches."""
    n_pts = max(n_matches, 1)
    kpts1 = np.random.rand(n_pts, 2) * 100
    kpts2 = np.random.rand(n_pts, 2) * 100
    matches = np.stack([np.arange(n_matches), np.arange(n_matches)], axis=1)
    K1, K2 = _intrinsics()
    R = np.eye(3)
    t = np.array([1.0, 0.0, 0.0])
    return (("a.png", "b.png"), matches, kpts1, kpts2, K1, K2, R, t)


def test_estimate_pose_returns_none_for_too_few_points():
    K1, K2 = _intrinsics()
    pts = np.zeros((3, 2))  # fewer than the 5 the essential matrix needs
    assert estimate_pose(pts, pts, K1, K2, norm_thresh=1.0) is None


def test_too_few_matches_returns_failure_sentinel():
    data = _pair_data(n_matches=3)
    assert process_pose_estimation(data, th=0.5) == FAILURE_RESULT


def test_none_pose_handled_explicitly_not_via_exception(monkeypatch, caplog):
    # estimate_pose can legitimately return None (degenerate geometry).
    # The caller must handle that explicitly and log it, not crash-then-swallow.
    monkeypatch.setattr(utils_benchmark, "estimate_pose", lambda *a, **k: None)
    data = _pair_data(n_matches=10)
    with caplog.at_level(logging.WARNING):
        result = process_pose_estimation(data, th=0.5)
    assert result == FAILURE_RESULT
    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "a failed pose estimation must be logged at WARNING or above"
    )


def test_unexpected_exception_is_logged_at_warning(monkeypatch, caplog):
    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(utils_benchmark, "estimate_pose", _boom)
    data = _pair_data(n_matches=10)
    with caplog.at_level(logging.WARNING):
        result = process_pose_estimation(data, th=0.5)
    assert result == FAILURE_RESULT
    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "an unexpected error must surface at WARNING, not be hidden at debug"
    )


def test_summarize_reports_failed_pair_count(caplog):
    # Aggregation must count and surface how many pairs failed.
    from benchmarks_2D.benchmark_parallel import Benchmark

    results = [
        ("a", "b", 1.0, 1.0, 2.0, 50),  # ok
        ("c", "d", 180, 180, 180, 0),  # failed
        ("e", "f", 180, 180, 180, 0),  # failed
    ]
    with caplog.at_level(logging.WARNING):
        out = Benchmark._summarize_pose_results(results)
    assert out["unregistered_pairs"] == 2
    assert out["total_pairs"] == 3
    assert any("2" in r.getMessage() for r in caplog.records), (
        "the number of failed/unregistered pairs must be logged"
    )
