"""Utility functions for the HPatches benchmark (warping, stats, and result display)."""

import os
import cv2
import math
import json
import torch
import logging
import numpy as np
import pandas as pd
from torch import Tensor
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from common.geometry import filter_outside
from common.serialization import (  # noqa: F401  re-exported for hpatches_benchmark
    convert_numpy_types,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.info(
        "tqdm not found, you'll get no progress bars. Install it with `pip install tqdm`."
    )
    from benchmarks_2D.utils_benchmark import fake_tqdm as tqdm


def load_hpatches_in_memory(wrapper, root: str | Path, max_workers: int = 16) -> Dict:
    """
    Load HPatches into memory with multi-threaded I/O.
    Returns: dict[name] -> {"imgs": [6 np arrays], "homs": [6 (3x3) np arrays]}
    """
    root = Path(root)
    assert root.exists(), f"HPatches root not found: {root}"

    # Typical HPatches layout: <root>/<scene>/{1.ppm..6.ppm, H_1_1..H_1_6}
    scenes = sorted([p for p in root.iterdir() if p.is_dir()])

    def _load_one(scene: Path):
        """Load one scene's six images and their reference homographies."""
        imgs, homs = [], []
        for j in range(1, 7):
            img = wrapper.load_image(scene / f"{j}.ppm").cpu()
            H = (
                np.eye(3, dtype=np.float64)
                if j == 1
                else np.loadtxt(scene / f"H_1_{j}")
            )
            imgs.append(img)
            homs.append(H)
        return scene.name, {"imgs": imgs, "homs": homs}

    hpatches = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for name, pack in tqdm(
            ex.map(_load_one, scenes), total=len(scenes), desc="Loading hpatches"
        ):
            hpatches[name] = pack
    return hpatches


def warp_points(
    xy: torch.Tensor,
    H: torch.Tensor,
    img_shape: tuple[float, float] | torch.Tensor | np.ndarray | None = None,
    border: int = 0,
) -> torch.Tensor:
    """
    Warp 2D points using a homography.

    Args:
        xy: points in (x, y) order. Shape:
            - (N, 2) for single set, or
            - (B, N, 2) for batched sets
        H: homography(ies). Shape:
            - (3, 3) for single H, or
            - (B, 3, 3) for batched H
           If one of (xy or H) is batched and the other is single, the single one
           is broadcast across the batch.
        img_shape: if provided (H, W), points projected outside (with optional border)
                   are set to NaN.
        border: with img_shape, mark points within `border` pixels of the boundary as NaN.

    Returns:
        Projected points with the same rank as `xy`:
            - (N, 2) if input xy was (N, 2)
            - (B, N, 2) if input xy was (B, N, 2)
    """
    assert xy.ndim in (2, 3), f"xy must be (N,2) or (B,N,2), got {xy.shape}"
    assert H.ndim in (2, 3), f"H must be (3,3) or (B,3,3), got {H.shape}"
    if img_shape is not None:
        assert len(img_shape) == 2, "img_shape must be (H, W)"

    single_xy = xy.ndim == 2  # (N,2)
    single_H = H.ndim == 2  # (3,3)

    # Add batch dimension if needed
    xy_b = xy[None, ...] if single_xy else xy  # (B?,N,2)
    H_b = H[None, ...] if single_H else H  # (B?,3,3)

    # Broadcast batch if one side has B=1
    B_xy = xy_b.shape[0]
    B_H = H_b.shape[0]
    if B_xy != B_H:
        if B_xy == 1:
            xy_b = xy_b.expand(B_H, -1, -1)
        elif B_H == 1:
            H_b = H_b.expand(B_xy, -1, -1)
        else:
            raise AssertionError(
                f"Incompatible batch sizes: xy batch={B_xy}, H batch={B_H}"
            )

    # Compute homogeneous projection
    ones = torch.ones(
        (xy_b.shape[0], xy_b.shape[1], 1), dtype=xy_b.dtype, device=xy_b.device
    )
    xy_hom = torch.cat([xy_b, ones], dim=2).to(H_b.dtype)  # (B,N,3)
    xy_proj_hom = xy_hom @ H_b.to(xy_b.device).permute(0, 2, 1)  # (B,N,3)

    den = xy_proj_hom[:, :, 2:3].clamp(min=1e-8)  # numerical safety
    xy_proj = xy_proj_hom[:, :, 0:2] / den  # (B,N,2)

    if img_shape is not None:
        xy_proj = filter_outside(xy_proj, img_shape, border)  # (B,N,2)

    xy_proj = xy_proj.to(xy.dtype)

    # Return same rank as input xy
    return xy_proj[0] if single_xy else xy_proj


def compute_homography_corner_error(
    H0_1_GT: Tensor,
    H0_1_estimated: Tensor,
    img0_shape: tuple[int, int] | Tensor,
    img1_shape: tuple[int, int] | Tensor | None = None,
) -> Tensor:
    """compute the corner error in pixel between the GT homography and the estimated one projecting the corners of
        img0 and img1 (if provided)
    Args:
        H0_1_GT: the GT homography warping between img0 and img1
            Bx3x3
        H0_1_estimated: the estimated homography between img0 and img1
            Bx3x3
        img0_shape: the shape of img0 used to project the four corners
            (H, W)
        img1_shape: the shape of img1 used to project the four corners, if not provided only one direction is used
            (H, W)
    Returns:
        corner_error: the computed (symmetric if img1_shape is provided) corner error
            B
    """
    assert H0_1_GT.dim() == 3
    assert H0_1_estimated.dim() == 3
    assert H0_1_GT.shape == H0_1_estimated.shape
    assert len(img0_shape) == 2
    assert img1_shape is None or len(img1_shape) == 2
    B = H0_1_estimated.shape[0]
    device = H0_1_GT.device

    corners0 = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, img0_shape[0]],
            [img0_shape[1], img0_shape[0]],
            [img0_shape[1], 0.0],
        ],
        device=device,
    )[None, :, :].repeat(B, 1, 1)  # B,4,2

    # ? compute the corner error in the projected frame
    projected_corners0_GT = warp_points(corners0, H0_1_GT.to(torch.double))  # B,4,2
    projected_corners0 = warp_points(corners0, H0_1_estimated.to(torch.double))  # B,4,2
    corner_error = torch.norm(projected_corners0_GT - projected_corners0, dim=2).mean(1)
    if img1_shape is not None:
        corners1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, img1_shape[0]],
                [img1_shape[1], img1_shape[0]],
                [img1_shape[1], 0.0],
            ],
            device=device,
        )[None, :, :].repeat(B, 1, 1)  # B,4,2
        # ? compute the corner error in the projected frame
        projected_corners1_GT = warp_points(
            corners1, torch.inverse(H0_1_GT.to(torch.double))
        )  # B,4,2
        projected_corners1 = warp_points(
            corners1, torch.inverse(H0_1_estimated.to(torch.double))
        )  # B,4,2
        corner_error_1 = torch.norm(
            projected_corners1_GT - projected_corners1, dim=2
        ).mean(1)
        corner_error = 0.5 * (corner_error + corner_error_1)  # B

    return corner_error


def _invalid_corner_row(thr, n_matches, device):
    """Build the corner-error row used when no valid homography is available."""
    return {
        "valid_homography": False,
        "ransac_thr": float(thr),
        "corner_error": float("inf"),
        "homography": torch.ones((3, 3), device=device) * float("nan"),
        "mask": torch.zeros(n_matches, dtype=torch.bool, device=device),
        "n_matched_keypoints": n_matches,
        "n_ransac_inliers": 0,
    }


def _solve_corner_homography(
    thr,
    xy0_np,
    xy1_np,
    H_gt,
    img0_shape,
    img1_shape,
    mode,
    n_matches,
    ransac_max_iters,
    device,
):
    """RANSAC one homography at ``thr`` and return its corner-error result row."""
    H_est, mask = cv2.findHomography(
        xy0_np,
        xy1_np,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(thr),
        maxIters=int(ransac_max_iters),
        confidence=0.999999,
    )
    if H_est is None:
        return _invalid_corner_row(thr, n_matches, device)

    # Corner error between GT and estimated H
    H_est_t = torch.as_tensor(H_est, dtype=torch.float32, device=device)
    ce = compute_homography_corner_error(
        H_gt[None, ...],
        H_est_t[None, ...],
        img0_shape,
        img1_shape if mode == "symmetric" else None,
    )[0].item()

    mask_bool = torch.as_tensor(mask.reshape(-1).astype(bool), device=device)
    return {
        "valid_homography": True,
        "ransac_thr": float(thr),
        "corner_error": float(ce),
        "homography": H_est,  # keep as numpy for JSON friendliness
        "mask": mask_bool,
        "n_matched_keypoints": n_matches,
        "n_ransac_inliers": int(mask_bool.sum().item()),
    }


def compute_corner_error(
    xy0_matched: torch.Tensor,
    xy1_matched: torch.Tensor,
    H0_1: torch.Tensor,
    img0_shape: tuple[int, int] | torch.Tensor,
    img1_shape: tuple[int, int] | torch.Tensor,
    mode: str,
    ransac_homography_threshold: list[float],
    ransac_max_iters: int = 10_000,
    njobs: int = 1,
) -> list[dict]:
    """
    Estimate ONE homography per threshold using ALL matches (no chunking), then
    compute the homography corner error. Parallelized across thresholds with joblib.

    Returns a list[dict] with fields:
      - valid_homography (bool)
      - ransac_thr (float)
      - corner_error (float)
      - homography (np.ndarray or NaN tensor)
      - mask (torch.BoolTensor of inliers length n_matches)
      - n_matched_keypoints (int)
      - n_ransac_inliers (int)
    """
    assert xy0_matched.shape == xy1_matched.shape and xy0_matched.shape[1] == 2
    assert H0_1.shape == (3, 3)

    device = xy0_matched.device
    n_matches = int(xy0_matched.shape[0])

    # Not enough points
    if n_matches < 4:
        return [
            _invalid_corner_row(thr, n_matches, device)
            for thr in ransac_homography_threshold
        ]

    xy0_np = xy0_matched.detach().cpu().numpy()
    xy1_np = xy1_matched.detach().cpu().numpy()
    H_gt = H0_1.detach()

    def _solve_one(thr: float):
        """Solve the corner-error homography for a single RANSAC threshold."""
        return _solve_corner_homography(
            thr,
            xy0_np,
            xy1_np,
            H_gt,
            img0_shape,
            img1_shape,
            mode,
            n_matches,
            ransac_max_iters,
            device,
        )

    # Parallel across thresholds
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = Parallel(n_jobs=njobs, prefer="threads")(
        delayed(_solve_one)(thr) for thr in ransac_homography_threshold
    )
    return results


def find_distance_matrices_between_points_and_their_projections(
    xy0: Tensor, xy1: Tensor, xy0_proj: Tensor, xy1_proj: Tensor
) -> Tuple[Tensor, Tensor]:
    """find the mutual nearest neighbors between two sets of keypoints and their projections
    Args:
        xy0: first set of keypoints
            n0,2
        xy1: second set of keypoints
            n1,2
        xy0_proj: first set of projected keypoints
            n0_proj,2
        xy1_proj: second set of projected keypoints
            n1_proj,2
    Returns:
        dist0: distance matrix between xy0 and xy1_proj
            n0,n1
        dist1: distance matrix between xy0_proj and xy1
            n0,n1
    """

    assert xy0.ndim == 2 and xy1.ndim == 2, (
        f"xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}"
    )
    assert xy0_proj.ndim == 2 and xy1_proj.ndim == 2, (
        f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}"
    )

    # ? compute the distance between all the reprojected points
    # # ? low memory usage, slow but correct
    # dist0 = torch.cdist(xy0.to(torch.float), xy1_proj,         compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(torch.float), compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # ? high memory usage, fast and correct
    dist0 = (xy0[:, None, :] - xy1_proj[None, :, :]).norm(dim=2)  # n0,n1
    dist1 = (xy0_proj[:, None, :] - xy1[None, :, :]).norm(dim=2)  # n0,n1
    # # ? low memory usage, fast but non-deterministic
    # dist0 = torch.cdist(xy0.to(torch.float), xy1_proj)  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(torch.float))  # n0,n1
    dist0[dist0.isnan()] = float("+inf")
    dist1[dist1.isnan()] = float("+inf")
    return dist0, dist1


def compute_coverages(
    xy0: Tensor,
    xy1: Tensor,
    xy0_proj: Tensor,
    xy1_proj: Tensor,
    img0_shape: Tensor,
    img1_shape: Tensor,
    px_thrs: float | list[float],
    coverage_kernel_size: int,
) -> Tuple[float, float] | Tuple[Tensor, Tensor]:
    """Compute spatial inlier coverage and per-keypoint coverage at each pixel threshold.

    Args:
        xy0: keypoints in image 0 (n0, 2).
        xy1: keypoints in image 1 (n1, 2).
        xy0_proj: keypoints of image 0 projected into image 1 (NaN if outside).
        xy1_proj: keypoints of image 1 projected into image 0 (NaN if outside).
        img0_shape: shape (H, W) of image 0.
        img1_shape: shape (H, W) of image 1.
        px_thrs: pixel inlier threshold(s).
        coverage_kernel_size: max-pool kernel size used to dilate inlier locations.

    Returns:
        (coverage, coverage_per_kpt): scalars if a single threshold is given,
        otherwise tensors indexed by threshold.
    """
    assert xy0.ndim == 2 and xy1.ndim == 2, (
        f"xy0 and xy1 must be 2D tensors, got {xy0.shape} and {xy1.shape}"
    )
    assert xy0_proj.ndim == 2 and xy1_proj.ndim == 2, (
        f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.shape} and {xy1_proj.shape}"
    )
    device = xy0.device

    # ? number of keypoints that are in the overlap area
    n_xy0_overlap = int((~torch.isnan(xy0_proj[:, 0])).sum())
    n_xy1_overlap = int((~torch.isnan(xy1_proj[:, 0])).sum())

    dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
        xy0, xy1, xy0_proj, xy1_proj
    )

    if not isinstance(px_thrs, list):
        px_thrs = [px_thrs]

    coverage = torch.zeros(len(px_thrs), device=device)
    coverage_per_kpt = torch.zeros(len(px_thrs), device=device)
    for i, px_thr in enumerate(px_thrs):
        xy0_inlier = xy0[(dist1 <= px_thr).any(1)]
        xy1_inlier = xy1[(dist0 <= px_thr).any(0)]
        ij0 = torch.flip(xy0_inlier.to(torch.long), [-1])
        ij1 = torch.flip(xy1_inlier.to(torch.long), [-1])
        img0 = torch.zeros(img0_shape, device=device)
        img1 = torch.zeros(img1_shape, device=device)
        img0[ij0[:, 0], ij0[:, 1]] = 1
        img1[ij1[:, 0], ij1[:, 1]] = 1
        img0 = torch.max_pool2d(
            img0[None, None, ...],
            (coverage_kernel_size, coverage_kernel_size),
            stride=1,
            padding=coverage_kernel_size // 2,
        )[0, 0, ...]
        img1 = torch.max_pool2d(
            img1[None, None, ...],
            (coverage_kernel_size, coverage_kernel_size),
            stride=1,
            padding=coverage_kernel_size // 2,
        )[0, 0, ...]
        coverage[i] = 0.5 * (img0.mean() + img1.mean())

        if n_xy0_overlap > 0 and n_xy1_overlap > 0:
            coverage_per_kpt[i] = (
                1000 * coverage[i] / (0.5 * (n_xy0_overlap + n_xy1_overlap))
            )
        else:
            coverage_per_kpt[i] = 0.0

    coverage = coverage[0].item() if coverage.shape[0] == 1 else coverage
    coverage_per_kpt = (
        coverage_per_kpt[0].item()
        if coverage_per_kpt.shape[0] == 1
        else coverage_per_kpt
    )

    return coverage, coverage_per_kpt


def _is_nan(x):
    """Return True if ``x`` is NaN, tolerating floats, numpy/torch scalars, and strings."""
    # Fast paths for common scalar types
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, np.floating):
        return np.isnan(x)

    # Optional: if a torch tensor ever sneaks in
    try:
        import torch

        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return bool(torch.isnan(x))
    except Exception:
        pass

    # Fallbacks
    try:
        return bool(np.isnan(x))  # works for numpy scalars
    except TypeError:
        # Try interpreting "nan"/"NaN" strings, etc.
        try:
            return math.isnan(float(x))
        except Exception:
            return False


def _as_float2(x, device):
    """Coerce ``x`` to a float32 (N, 2) tensor on ``device`` (empty if None)."""
    if x is None:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    if torch.is_tensor(x):
        x = x.to(device=device, dtype=torch.float32)
    else:
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.reshape(-1, 2)
    return x


def _as_long2(matches, device):
    """Coerce ``matches`` to a long (M, 2) index tensor on ``device`` (empty if None)."""
    if matches is None:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    if torch.is_tensor(matches):
        m = matches.to(device=device, dtype=torch.long)
    else:
        m = torch.as_tensor(matches, dtype=torch.long, device=device)
    if m.ndim == 1:
        m = m.reshape(-1, 2)
    elif m.ndim == 2 and m.shape[1] != 2:
        if m.shape[0] == 2:
            m = m.T.contiguous()
        else:
            m = m.reshape(-1, 2)
    return m


def _safe_div(num, den):
    """Divide ``num`` by ``den``, returning 0.0 when the denominator is non-positive."""
    num = float(num)
    den = float(den)
    return (num / den) if den > 0.0 else 0.0


def _inside(shape_hw, xy: torch.Tensor) -> torch.Tensor:
    """Return mask of points inside image (h,w). Uses half-open range [0,w) and [0,h)."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if xy.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=xy.device)
    x, y = xy[:, 0], xy[:, 1]
    return (x >= 0) & (y >= 0) & (x < w) & (y < h)


def _overlap_cdist(src_w_vis, dst, n_overlap, device):
    """Pairwise distances from warped-visible source points to dst (empty-safe)."""
    if n_overlap and dst.shape[0]:
        return torch.cdist(src_w_vis, dst, p=2)
    return torch.empty((n_overlap, dst.shape[0]), dtype=torch.float32, device=device)


def _match_errors(matches, xy0, xy1, H0_1, H1_0, device):
    """Symmetric (max of both directions) reprojection error per proposed match."""
    if not matches.numel():
        return torch.zeros((0,), dtype=torch.float32, device=device)

    i0 = matches[:, 0].clamp(min=0, max=max(0, xy0.shape[0] - 1))
    i1 = matches[:, 1].clamp(min=0, max=max(0, xy1.shape[0] - 1))
    xy0_m = xy0[i0]
    xy1_m = xy1[i1]
    xy0_m_w = warp_points(xy0_m, H0_1)
    xy1_m_w = warp_points(xy1_m, H1_0)
    err_0to1 = (
        torch.linalg.norm(xy0_m_w - xy1_m, dim=1)
        if xy1_m.shape[0]
        else torch.zeros(0, device=device)
    )
    err_1to0 = (
        torch.linalg.norm(xy1_m_w - xy0_m, dim=1)
        if xy0_m.shape[0]
        else torch.zeros(0, device=device)
    )
    return torch.maximum(err_0to1, err_1to0)


def _coverage_rows(compute_coverage, n_xy0_overlap, n_xy1_overlap, xy0, xy1):
    """Mean overlap-coverage row (empty list when coverage is disabled)."""
    if not compute_coverage:
        return []
    cov0 = float(n_xy0_overlap) / float(max(1, xy0.shape[0]))
    cov1 = float(n_xy1_overlap) / float(max(1, xy1.shape[0]))
    cov = 0.5 * (cov0 + cov1)
    cov_pk = 0.5 * (
        _safe_div(n_xy0_overlap, xy0.shape[0]) + _safe_div(n_xy1_overlap, xy1.shape[0])
    )
    return [{"coverage": cov, "coverage_per_kpt": cov_pk}]


def _overlap_geometry(xy0, xy1, H0_1, H1_0, img0_shape, img1_shape, device):
    """Warp keypoints across the GT homography and return overlap distances.

    Returns:
        (d01, d10, n_xy0_overlap, n_xy1_overlap): distances from each image's
        warped-visible keypoints to the other's keypoints, plus overlap counts.
    """
    xy0_w = warp_points(xy0, H0_1)
    xy1_w = warp_points(xy1, H1_0)

    m0 = _inside(img1_shape, xy0_w)
    m1 = _inside(img0_shape, xy1_w)
    xy0_w_vis = xy0_w[m0]
    xy1_w_vis = xy1_w[m1]

    n_xy0_overlap = int(m0.sum().item())
    n_xy1_overlap = int(m1.sum().item())

    d01 = _overlap_cdist(xy0_w_vis, xy1, n_xy0_overlap, device)
    d10 = _overlap_cdist(xy1_w_vis, xy0, n_xy1_overlap, device)
    return d01, d10, n_xy0_overlap, n_xy1_overlap


def compute_matching_stats_homography(
    xy0,
    xy1,
    H0_1,  # ground-truth homography 1->2
    img0_shape,
    img1_shape,
    mode: str = "hpatches",
    matches=None,  # proposed matches (M,2) [idx0, idx1]
    px_thrs: List[float] = [1, 2, 3, 4, 5],
    ransac_thresholds: List[float] | None = None,  # Add this parameter
    compute_coverage: bool = True,
    coverage_kernel_size: int = 9,
    evaluate_corner_error: bool = True,
    evaluate_corner_error_keypoints: bool = False,
    device: str = "cpu",
    njobs: int = 1,
):
    """Compute per-threshold matching stats and RANSAC homography corner-error rows for a pair.

    Args:
        xy0: keypoints in image 0.
        xy1: keypoints in image 1.
        H0_1: ground-truth homography mapping image 0 to image 1.
        img0_shape: shape (H, W) of image 0.
        img1_shape: shape (H, W) of image 1.
        mode: corner-error mode passed downstream (e.g. "hpatches"/"symmetric").
        matches: proposed matches (M, 2) as [idx0, idx1].
        px_thrs: pixel thresholds for repeatability/accuracy/score.
        ransac_thresholds: RANSAC reprojection thresholds for homography estimation.
        compute_coverage: whether to compute coverage rows.
        coverage_kernel_size: kernel size for coverage dilation.
        evaluate_corner_error: whether to estimate homographies and corner errors.
        evaluate_corner_error_keypoints: unused flag kept for signature compatibility.
        device: torch device for the computation.
        njobs: number of parallel jobs (unused here; inner calls run single-threaded).

    Returns:
        (list_stats, list_hstats, list_cov): threshold stat rows, homography
        evaluation rows, and coverage rows.
    """
    device = torch.device(device)
    xy0 = _as_float2(xy0, device)
    xy1 = _as_float2(xy1, device)
    H0_1 = torch.as_tensor(H0_1, dtype=torch.float32, device=device)
    H1_0 = torch.inverse(H0_1)
    matches = _as_long2(matches, device)

    d01, d10, n_xy0_overlap, n_xy1_overlap = _overlap_geometry(
        xy0, xy1, H0_1, H1_0, img0_shape, img1_shape, device
    )

    match_err = _match_errors(matches, xy0, xy1, H0_1, H1_0, device)
    list_cov = _coverage_rows(compute_coverage, n_xy0_overlap, n_xy1_overlap, xy0, xy1)

    list_stats = _homography_threshold_rows(
        px_thrs,
        d01,
        d10,
        match_err,
        n_xy0_overlap,
        n_xy1_overlap,
        list_cov,
        compute_coverage,
        device,
    )
    list_hstats = _homography_eval_rows(
        matches,
        xy0,
        xy1,
        H0_1,
        img0_shape,
        img1_shape,
        mode,
        ransac_thresholds,
        px_thrs,
        evaluate_corner_error,
    )
    return list_stats, list_hstats, list_cov


def _homography_threshold_rows(
    px_thrs,
    d01,
    d10,
    match_err,
    n_xy0_overlap,
    n_xy1_overlap,
    list_cov,
    compute_coverage,
    device,
):
    """Per-threshold repeatability / accuracy / precision / recall rows."""
    list_stats = []
    for px_thr in px_thrs:
        rep0 = (
            (d01.min(dim=1).values <= px_thr).float().mean().item()
            if d01.numel()
            else 0.0
        )
        rep1 = (
            (d10.min(dim=1).values <= px_thr).float().mean().item()
            if d10.numel()
            else 0.0
        )
        repeatability = 0.5 * (rep0 + rep1)

        n_prop = int(match_err.numel())
        good = (
            (match_err <= px_thr)
            if n_prop
            else torch.zeros(0, dtype=torch.bool, device=device)
        )
        n_good = int(good.sum().item())

        n_gt0 = int(
            ((d01.min(dim=1).values <= px_thr).sum().item()) if d01.numel() else 0
        )
        n_gt1 = int(
            ((d10.min(dim=1).values <= px_thr).sum().item()) if d10.numel() else 0
        )
        n_gt_corr = 0.5 * (n_gt0 + n_gt1)

        matching_score = 0.5 * (
            _safe_div(n_good, n_xy0_overlap) + _safe_div(n_good, n_xy1_overlap)
        )

        row = {
            "thr": float(px_thr),
            "repeatability": float(repeatability),
            "repeatability_mnn": float(repeatability),
            "matching_accuracy": float(_safe_div(n_good, n_prop)),
            "matching_score": float(matching_score),
            "precision": float(_safe_div(n_good, n_prop)),
            "recall": float(_safe_div(n_good, n_gt_corr)),
            "n_matches_proposed": float(n_prop),
            "n_inliers_nn_GT": float(n_good),
        }
        if compute_coverage and len(list_cov):
            row["coverage"] = float(list_cov[0]["coverage"])
            row["coverage_per_kpt"] = float(list_cov[0]["coverage_per_kpt"])
        list_stats.append(row)
    return list_stats


def _homography_eval_rows(
    matches,
    xy0,
    xy1,
    H0_1,
    img0_shape,
    img1_shape,
    mode,
    ransac_thresholds,
    px_thrs,
    evaluate_corner_error,
):
    """RANSAC homography corner-error rows (fills list_hstats)."""
    list_hstats = []
    if not evaluate_corner_error:
        return list_hstats

    # Use custom RANSAC thresholds if provided, otherwise a default set.
    if ransac_thresholds is None:
        ransac_thresholds = [0.125, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]

    if matches.numel() >= 8:
        i0 = matches[:, 0].clamp(min=0, max=max(0, xy0.shape[0] - 1))
        i1 = matches[:, 1].clamp(min=0, max=max(0, xy1.shape[0] - 1))
        ce_list = compute_corner_error(
            xy0_matched=xy0[i0],
            xy1_matched=xy1[i1],
            H0_1=H0_1,
            img0_shape=img0_shape,
            img1_shape=img1_shape,
            mode=mode,
            ransac_homography_threshold=ransac_thresholds,
            ransac_max_iters=10_000,
            njobs=1,
        )
        for ce in ce_list:
            list_hstats.append(
                {
                    "ransac_thr": float(ce["ransac_thr"]),
                    "corner_error": float(ce["corner_error"]),
                    "valid_homography": bool(ce["valid_homography"]),
                    "n_ransac_inliers": int(ce["n_ransac_inliers"]),
                }
            )
    else:
        # If not enough matches, use px_thrs for consistency in output
        for thr in px_thrs:
            list_hstats.append(
                {
                    "ransac_thr": float(thr),
                    "corner_error": float("inf"),
                    "valid_homography": False,
                    "n_ransac_inliers": 0,
                }
            )
    return list_hstats


_HPATCHES_METRIC_NAMES = [
    "repeatability",
    "repeatability_mnn",
    "n_keypoints",
    "coverage",
    "coverage_per_kpt",
    "matching_accuracy",
    "matching_score",
    "precision",
    "recall",
    "n_matches_proposed",
    "n_inliers_nn_GT",
]


def _compute_one_pair_stats(
    folder_name, j, hpatches, keypoints, matches, max_kpts, px_thrs, ransac_thresholds
):
    """Evaluate a single (sequence, pair) and return ``(stats_df, homography_df)``."""
    imgs = hpatches[folder_name]["imgs"]
    homs = hpatches[folder_name]["homs"]

    img0_shape = imgs[0].shape[-2:]
    img1_shape = imgs[j].shape[-2:]
    H01 = torch.as_tensor(homs[j], dtype=torch.float32)

    xy0 = keypoints[folder_name][0]
    xy1 = keypoints[folder_name][j]
    if max_kpts < 999_999:
        xy0 = xy0[:max_kpts]
        xy1 = xy1[:max_kpts]

    pair_key = f"1_{j + 1}"
    m01 = matches.get(folder_name, {}).get(pair_key, None)

    # Inner call MUST be single-threaded to avoid oversubscription
    list_stats, list_hstats, _ = compute_matching_stats_homography(
        xy0=xy0,
        xy1=xy1,
        H0_1=H01,
        img0_shape=img0_shape,
        img1_shape=img1_shape,
        mode="hpatches",
        matches=m01,
        px_thrs=px_thrs,
        ransac_thresholds=ransac_thresholds,
        compute_coverage=True,
        coverage_kernel_size=9,
        evaluate_corner_error=True,
        evaluate_corner_error_keypoints=False,
        device="cpu",
        njobs=1,
    )

    sdf = pd.DataFrame(list_stats)
    hdf = pd.DataFrame(list_hstats)

    # Annotate with metadata
    seq_type = folder_name[0] if folder_name else "?"
    pair_name = f"1-{j + 1}"
    n_kpts = 0.5 * (len(xy0) + len(xy1))
    for df in (sdf, hdf):
        if len(df):
            df["type"] = seq_type
            df["scene"] = folder_name
            df["pair"] = pair_name
            df["n_keypoints"] = n_kpts

    return sdf, hdf


def _aggregate_matching_stats(stats_df):
    """Mean/median of matching metrics by (thr, type), with an 'overall' partition."""
    if len(stats_df):
        stats_df_overall = stats_df.copy()
        stats_df_overall["type"] = "overall"
        stats_df_aug = pd.concat([stats_df, stats_df_overall], ignore_index=True)
    else:
        stats_df_aug = stats_df.copy()

    if not len(stats_df_aug):
        return pd.DataFrame()

    mean_renaming = {n: f"mean_{n}" for n in _HPATCHES_METRIC_NAMES}
    median_renaming = {n: f"median_{n}" for n in _HPATCHES_METRIC_NAMES}
    numeric_cols = stats_df_aug.select_dtypes(include="number").columns
    mean_stats_df = stats_df_aug.groupby(["thr", "type"], as_index=False)[
        numeric_cols
    ].mean()
    mean_stats_df.rename(columns=mean_renaming, inplace=True)
    median_stats_df = stats_df_aug.groupby(["thr", "type"], as_index=False)[
        numeric_cols
    ].median()
    median_stats_df.rename(columns=median_renaming, inplace=True)
    return pd.merge(mean_stats_df, median_stats_df, how="outer")


def _aggregate_homography_accuracy(stats_homography_df):
    """Homography-accuracy proportions per (ransac_thr, type) across accuracy thresholds."""
    if not len(stats_homography_df):
        return pd.DataFrame(
            columns=["ransac_thr", "type", "homography_accuracy", "accuracy_thr"]
        )

    stats_h_overall = stats_homography_df.copy()
    stats_h_overall["type"] = "overall"
    stats_h_aug = pd.concat([stats_homography_df, stats_h_overall], ignore_index=True)

    aggregated = pd.DataFrame()
    homography_accuracy_thrs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for acc_thr in homography_accuracy_thrs:
        tmp = stats_h_aug.copy()
        # Boolean then mean() -> proportion
        tmp["homography_accuracy"] = tmp["corner_error"] <= acc_thr
        stats_at_thr = (
            tmp.groupby(["ransac_thr", "type"])["homography_accuracy"]
            .mean()
            .reset_index()
        )
        stats_at_thr["accuracy_thr"] = float(acc_thr)
        aggregated = pd.concat([aggregated, stats_at_thr], ignore_index=True)
    return aggregated


def compute_matching_stats(
    keypoints: Dict,
    matches: Dict,
    hpatches: Dict,
    max_kpts: int = 999_999,
    px_thrs: float | list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ransac_thresholds: list[float] | None = None,
    njobs: int = 16,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parallel per-(sequence, pair) evaluation with joblib.

    Returns (stats_df, aggregated_df, stats_homography_df, aggregated_homography_accuracy_df)
    to match what hpatches_benchmark.py expects.
    """
    if not isinstance(px_thrs, list):
        px_thrs = [px_thrs]

    # Build all work items first: (folder_name, j) for pairs (1,2)..(1,6)
    work: List[tuple[str, int]] = []
    for folder_name, pack in hpatches.items():
        for j in range(1, min(6, len(pack["imgs"]))):
            work.append((folder_name, j))

    # Using threads is faster here since each job is lightweight
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = Parallel(n_jobs=njobs, prefer="threads")(
        delayed(_compute_one_pair_stats)(
            folder, j, hpatches, keypoints, matches, max_kpts, px_thrs, ransac_thresholds
        )
        for (folder, j) in tqdm(work, desc="Computing pairs")
    )

    # Concatenate per-pair outputs
    stats_df_list, stats_h_df_list = [], []
    for sdf, hdf in results:
        if len(sdf):
            stats_df_list.append(sdf)
        if len(hdf):
            stats_h_df_list.append(hdf)

    stats_df = (
        pd.concat(stats_df_list, ignore_index=True) if stats_df_list else pd.DataFrame()
    )
    stats_homography_df = (
        pd.concat(stats_h_df_list, ignore_index=True)
        if stats_h_df_list
        else pd.DataFrame()
    )

    # Return exactly what hpatches_benchmark.py expects
    return (
        stats_df,
        _aggregate_matching_stats(stats_df),
        stats_homography_df,
        _aggregate_homography_accuracy(stats_homography_df),
    )


def display_hpatches_results_(
    results_file="hpatches/results/results.json",
    partition="overall",
    method=None,
    ths=None,  # Allow None as default
    tostring=True,
):
    """
    Display HPatches benchmark results in a formatted DataFrame.

    Args:
        results_file: Path to the results JSON file
        partition: Which partition to display ('overall', 'i', 'v')
        method: Specific method to display (if None, displays all methods)
        ths: List of thresholds to display (if None, auto-detect from results)
        tostring: If True, print the DataFrame as a string

    Returns:
        pandas.DataFrame: Formatted results table
    """
    # Load results
    if isinstance(results_file, str):
        results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        print("No results found in the file.")
        return pd.DataFrame()

    # Filter by specific method if provided
    if method is not None:
        if method in data:
            data = {method: data[method]}
        else:
            print(f"Method '{method}' not found in results.")
            return pd.DataFrame()

    # Auto-detect thresholds if not provided
    if ths is None:
        ths = []
        for method_key, results in data.items():
            if partition in results:
                partition_data = results[partition]
                for key in partition_data.keys():
                    if key.startswith("repeatability_"):
                        thr = key.split("_")[-1]
                        try:
                            thr_val = int(float(thr))
                            if thr_val not in ths:
                                ths.append(thr_val)
                        except ValueError:
                            continue
        ths = sorted(ths) if ths else [1, 2, 3]  # Default fallback

    # Create DataFrame
    rows = []

    for method_key, results in data.items():
        if partition not in results:
            continue

        partition_data = results[partition]

        # Base row info
        if "+" in method_key:
            method_name, custom_desc_part = method_key.split("+", 1)
            custom_desc = custom_desc_part.split("_")[0]
        else:
            method_name = method_key.split("_")[0]
            custom_desc = ""

        row = {
            "Method": method_name,
            "Custom Desc": custom_desc,
        }

        # Add metrics for all thresholds - GROUPED BY METRIC TYPE
        for thr in ths:
            # Repeatability
            rep_key = f"repeatability_{thr}"
            if rep_key in partition_data:
                rep_data = partition_data[rep_key]
                row[f"Rep@{thr}"] = f"{rep_data.get('mean', 0.0) * 100:.1f}"

        for thr in ths:
            # Matching Accuracy
            ma_key = f"matching_accuracy_{thr}"
            if ma_key in partition_data:
                ma_data = partition_data[ma_key]
                row[f"MA@{thr}"] = f"{ma_data.get('mean', 0.0) * 100:.1f}"

        for thr in ths:
            # Matching Score
            ms_key = f"matching_score_{thr}"
            if ms_key in partition_data:
                ms_data = partition_data[ms_key]
                row[f"MS@{thr}"] = f"{ms_data.get('mean', 0.0) * 100:.1f}"

        for thr in ths:
            # Homography Accuracy
            ha_key = f"homography_accuracy_{thr}"
            if ha_key in partition_data:
                ha_data = partition_data[ha_key]
                row[f"HA@{thr}"] = f"{ha_data.get('mean', 0.0) * 100:.1f}"

        rows.append(row)

    if not rows:
        print(f"No results found for partition '{partition}'")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by method name for consistent ordering
    if len(df) > 1:
        df = df.sort_values(["Method", "Custom Desc"], ascending=[True, True])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    if tostring:
        print(df.to_string(index=False))
    return df


def _detect_repeatability_thresholds(data, partition):
    """Auto-detect integer thresholds from ``repeatability_*`` keys in the results."""
    ths_detected: List[int] = []
    for _, results in data.items():
        if partition not in results:
            continue
        for key in results[partition]:
            if not key.startswith("repeatability_"):
                continue
            try:
                thr_val = int(float(key.split("_")[-1]))
            except ValueError:
                continue
            if thr_val not in ths_detected:
                ths_detected.append(thr_val)
    return sorted(ths_detected) if ths_detected else [1, 2, 3]


def _build_hpatches_rows(data, partition, ths):
    """Build percentage-formatted display rows (one per method)."""
    metric_specs = [
        ("Rep", "repeatability"),
        ("MA", "matching_accuracy"),
        ("MS", "matching_score"),
        ("HA", "homography_accuracy"),
    ]
    rows: List[Dict[str, str]] = []
    for method_key, results in data.items():
        if partition not in results:
            continue
        partition_data = results[partition]

        # Method / Custom Desc parsing
        if "+" in method_key:
            method_name, custom_desc_part = method_key.split("+", 1)
            custom_desc = custom_desc_part.split("_")[0]
        else:
            method_name = method_key.split("_")[0]
            custom_desc = ""

        row: Dict[str, str] = {"Method": method_name, "Desc": custom_desc}
        for label, metric in metric_specs:
            for thr in ths:
                key = f"{metric}_{thr}"
                if key in partition_data:
                    mean = partition_data[key].get("mean", 0.0)
                    row[f"{label}@{thr}"] = f"{mean * 100:.1f}"
        rows.append(row)
    return rows


def _print_hpatches_table(rows, ths):
    """Pretty-print the results table to stdout."""
    fixed_cols = ["Method", "Custom Desc"]
    metric_cols_order: List[str] = []
    for label in ("Rep", "MA", "MS", "HA"):
        for thr in ths:
            metric_cols_order.append(f"{label}@{thr}")

    # Keep only metrics that appear at least once
    present_metrics = [c for c in metric_cols_order if any(c in r for r in rows)]
    cols = fixed_cols + present_metrics

    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)


def display_hpatches_results(
    results_file: Union[str, Path] = "hpatches/results/results.json",
    partition: str = "overall",
    method: Optional[str] = None,
    ths: Optional[List[int]] = None,
    tostring: bool = True,
) -> List[Dict[str, str]]:
    """
    Display HPatches benchmark results without pandas.

    Args:
        results_file: Path to the results JSON file
        partition: Which partition to display ('overall', 'i', 'v')
        method: Specific method to display (if None, displays all methods)
        ths: List of thresholds to display (if None, auto-detect from results)
        tostring: If True, print the table as a string

    Returns:
        List[Dict[str, str]]: Formatted results rows
    """
    # Load results
    if isinstance(results_file, str):
        results_file = Path(results_file)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        if tostring:
            print("No results found in the file.")
        return []

    # Filter by specific method if provided
    if method is not None:
        if method in data:
            data = {method: data[method]}
        else:
            if tostring:
                print(f"Method '{method}' not found in results.")
            return []

    if ths is None:
        ths = _detect_repeatability_thresholds(data, partition)

    rows = _build_hpatches_rows(data, partition, ths)
    if not rows:
        if tostring:
            print(f"No results found for partition '{partition}'")
        return []

    # Sort by Method then Custom Desc
    rows.sort(key=lambda r: (r.get("Method", ""), r.get("Custom Desc", "")))

    if tostring:
        _print_hpatches_table(rows, ths)

    return rows
