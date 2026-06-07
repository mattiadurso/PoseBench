"""3D scene pose-estimation benchmark comparing COLMAP reconstructions."""

from typing import Any, Optional, Sequence

import numpy as np

from common.geometry import is_torch


def to_numpy(vector: Any) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    """
    if is_torch(vector):
        return vector.detach().cpu().numpy()
    return vector


def evaluate_R_err(R_gt: np.ndarray, R: np.ndarray, deg: bool = True) -> float:
    """Compute the rotation error between two rotation matrices.

    Args:
        R_gt: Ground-truth rotation matrix (3x3).
        R: Predicted rotation matrix (3x3).
        deg: If True, return the error in degrees, otherwise in radians.

    Returns:
        The rotation error as a Python float.
    """
    eps = 1e-15

    # Make and normalize the quaternions.
    q = rotmat2qvec(R)
    q_gt = rotmat2qvec(R_gt)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    # Relative Rotation Angle in radians. Equivalant to acos(trace(R)*.5) with R = R_gt*R^T but more stable.
    loss_q = np.maximum(
        eps, (1.0 - np.inner(q, q_gt) ** 2)
    )  # Max to void NaNs, always > 0 due to **2.
    err_q = np.arccos(1 - 2 * loss_q)

    if deg:
        err_q = np.rad2deg(err_q)  # rad*180/np.pi

    if np.sum(np.isnan(err_q)):
        raise ValueError(
            "NaN encountered while computing the rotation error; check the input poses."
        )

    return err_q.item()


def evaluate_t_err(t_gt: np.ndarray, t: np.ndarray, deg: bool = True) -> float:
    """Compute the angular error between two translation vectors.

    Args:
        t_gt: Ground-truth translation vector.
        t: Predicted translation vector.
        deg: If True, return the error in degrees, otherwise in radians.

    Returns:
        The translation (angular) error as a Python float.
    """
    t_gt = to_numpy(t_gt)
    t = to_numpy(t)
    # Flatten
    t = t.flatten()
    t_gt = t_gt.flatten()
    eps = 1e-15

    # Equivalent to arccos(cosine_sim(t,t_gt))
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.inner(t, t_gt) ** 2))  # Max to void NaNs
    err_t = np.arccos(np.sqrt(1 - loss_t))
    # err_t = np.arccos(np.clip(np.inner(t,t_gt), -1.0, 1.0)) # Equivalent to above

    if np.sum(np.isnan(err_t)):
        raise ValueError(
            "NaN encountered while computing the translation error; check the inputs."
        )

    if deg:
        err_t = np.rad2deg(err_t)  # rad*180/np.pi

    return err_t.item()


def evaluate_R_t(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    deg: bool = True,
) -> np.ndarray:
    """
    Evaluate the rotation and translation errors between two poses. From IMC2020.
    Args:
        R_gt: Ground truth relative rotation matrix.
        t_gt: Ground truth relative translation vector.
        R:    Predicted relative rotation matrix.
        t:    Predicted relative translation vector.
    Returns:
        err_q: Rotation error in radians.
        err_t: Translation error in radians.
    """
    err_q = evaluate_R_err(R_gt, R, deg=deg)
    err_t = evaluate_t_err(t_gt, t, deg=deg)

    return np.stack([err_q, err_t])


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """From COLMAP implementation."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """From COLMAP implementation."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def compute_recall(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the recall for the errors. From Pixel-Perfect SfM.
    Args:
        errors: numpy array or errors.
    Returns:
        errors: sorted errors.
        recall: recall for each.

    """
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements  # cumsum accuracy?
    return errors, recall


def compute_AUC(
    errors: np.ndarray,
    thresholds: Sequence[float],
    min_error: Optional[float] = None,
) -> list[float]:
    """
    Compute the AUC for one array of errors. From Pixel-Perfect SfM.
    Args:
        errors: numpy array or errors.
        thresholds: list of thresholds for the AUC computation.
        min_error: minimum error to consider.
    Returns:
        aucs: list with the AUC values for each threshold.
    Note:
        - It is computed as the defined integral of the recall over the error.
        - This is NOT the same metric as ``pose_auc`` in
          ``benchmarks_2D/utils_benchmark.py``. This variant (Pixel-Perfect SfM)
          scales results to ``[0, 100]``, supports a ``min_error`` floor, and uses
          ``searchsorted(..., side="right")``. They are intentionally distinct.
    """
    n = len(errors)

    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / n
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:  # [1,3,5]
        last_index = np.searchsorted(
            errors, t, side="right"
        )  # index of the first element >= t
        r = np.r_[recall[:last_index], recall[last_index - 1]]  # error < t
        e = np.r_[errors[:last_index], t]
        auc = np.trapezoid(r, x=e) / t  # ?
        aucs.append(auc * 100)
    return aucs
