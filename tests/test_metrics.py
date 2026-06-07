"""Tests for the pose-error metrics in benchmarks_3D.utils_benchmark_pose."""

import math

import numpy as np

from benchmarks_3D.utils_benchmark_pose import compute_AUC, evaluate_R_t


def _rot_z(theta):
    """Return a rotation matrix of theta radians about the z-axis."""
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_evaluate_R_t_identity_is_zero_error():
    """evaluate_R_t returns near-zero errors for identical pose inputs."""
    R = np.eye(3)
    t = np.array([1.0, 0.0, 0.0])
    errs = evaluate_R_t(R, t, R, t)
    assert errs.shape == (2,)
    # An eps floor inside the arccos keeps errors near (but not exactly) zero.
    assert np.allclose(errs, 0.0, atol=1e-4)


def test_evaluate_R_t_known_rotation_error_degrees():
    """evaluate_R_t reports the correct rotation error in degrees."""
    R_gt = np.eye(3)
    R = _rot_z(math.radians(30.0))
    t = np.array([1.0, 0.0, 0.0])
    err_q, err_t = evaluate_R_t(R_gt, t, R, t, deg=True)
    assert math.isclose(err_q, 30.0, abs_tol=1e-3)
    assert math.isclose(err_t, 0.0, abs_tol=1e-4)


def test_evaluate_R_t_translation_error_is_angular():
    """evaluate_R_t measures translation error as the angle between vectors."""
    R = np.eye(3)
    t_gt = np.array([1.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 0.0])  # orthogonal -> 90 deg
    err_q, err_t = evaluate_R_t(R, t_gt, R, t, deg=True)
    assert math.isclose(err_q, 0.0, abs_tol=1e-4)
    assert math.isclose(err_t, 90.0, abs_tol=1e-3)


def test_compute_AUC_perfect_errors_are_full_score():
    """compute_AUC returns 100 per threshold when all errors are zero."""
    # All errors are zero -> recall saturates immediately -> AUC == 100 per threshold.
    errors = np.zeros(10)
    aucs = compute_AUC(errors, thresholds=[1, 3, 5])
    assert len(aucs) == 3
    assert all(math.isclose(a, 100.0, abs_tol=1e-6) for a in aucs)


def test_compute_AUC_monotonic_in_threshold():
    """compute_AUC is non-decreasing in threshold and bounded in [0, 100]."""
    rng = np.random.default_rng(0)
    errors = rng.uniform(0, 5, size=200)
    aucs = compute_AUC(errors, thresholds=[1, 3, 5])
    # Larger thresholds admit more recall, so AUC is non-decreasing.
    assert aucs[0] <= aucs[1] + 1e-6
    assert aucs[1] <= aucs[2] + 1e-6
    assert all(0.0 <= a <= 100.0 + 1e-6 for a in aucs)
