import cv2
import h5py
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from wrappers.wrapper import MethodWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fake_tqdm(x: Any, **kwargs: Any) -> Any:
    return x


def parse_poses(poses_file: Union[str, Path], benchmark_name: str) -> dict:
    """Parse poses from a given file based on the benchmark format."""
    if benchmark_name in [
        "megadepth1500",
        "graz4k",
        "megadepth_view",
        "megadepth_air2ground",
        "terrasky3d",
    ]:
        return parse_md1500_poses(poses_file)
    elif benchmark_name in ["scannet1500"]:
        # Implement parse_scannet1500_poses if needed
        raise NotImplementedError("Parsing for scannet1500 not implemented yet.")
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")


def load_depth(
    depth_path: Union[str, Path], scale_factor: float, target: torch.Tensor
) -> torch.Tensor:
    """Load depth data from a given file."""
    device, dtype = target.device, target.dtype
    depth = torch.tensor(h5py.File(depth_path, "r")["depth"][()], device=device).float()
    # crop multiple of 16 for compatibility with disk
    H, W = depth.shape
    depth = depth[: (H // 16) * 16, : (W // 16) * 16]  # same as image in img_from_numpy

    if scale_factor != 1.0:  # mostly for GHR, with md is must be 1.0
        depth = F.interpolate(
            depth[None, None],
            scale_factor=scale_factor,
            mode="nearest",
            align_corners=False,
        )[0, 0]

    return depth.to(device=device, dtype=dtype)


def parse_md1500_poses(poses_file: Union[str, Path]) -> dict:
    with open(poses_file, "r") as f:
        lines = f.readlines()

    view_dict = {}
    for line in lines:
        if line[0] == "#":
            continue
        elems = line.split()
        img_path = elems[0]
        R = np.array(elems[1:10], dtype=float).reshape(3, 3)
        t = np.array(elems[10:13], dtype=float).reshape(3, 1)
        camera_model = elems[13]
        image_size = np.array(elems[14:16], dtype=int)
        K_ = np.array(elems[16:], dtype=float)  # fx, fy, cx, cy
        K = np.eye(3)
        K[0, 0] = K_[0]
        K[1, 1] = K_[1]
        K[0, 2] = K_[2]
        K[1, 2] = K_[3]
        P = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1.0]).reshape(1, 4)))

        view_dict[img_path] = {
            "img_path": img_path,
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
            "P": torch.from_numpy(P),
            "camera_model": camera_model,
            "image_size": image_size,
        }

    return view_dict


def _solve_pair_pose(
    matches: Union[np.ndarray, torch.Tensor],
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    th: float,
) -> Optional[tuple]:
    """Solve the relative pose for one matched pair.

    Returns ``(R, t, inlier_mask)`` from :func:`estimate_pose`, or ``None`` for
    degenerate geometry. Matches are shuffled to decorrelate RANSAC sampling order.
    """
    matched_kpts1 = kpts1[matches[:, 0]]
    matched_kpts2 = kpts2[matches[:, 1]]

    shuffling = np.random.permutation(len(matched_kpts1))
    matched_kpts1 = matched_kpts1[shuffling]
    matched_kpts2 = matched_kpts2[shuffling]

    norm_threshold = th / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
    return estimate_pose(
        matched_kpts1, matched_kpts2, K1, K2, norm_threshold, conf=0.99999
    )


def process_pose_estimation(
    pair_matches_data: tuple, th: float, seed: int = 0
) -> tuple:
    """Estimate the relative pose for a single image pair.

    Returns the per-pair result tuple ``(img1, img2, e_t, e_R, e_pose, inliers)``.
    A pair that cannot be solved yields the 180-degree sentinel result so the
    benchmark keeps going; such failures are logged at WARNING so they stay
    visible rather than being silently swallowed.
    """
    fix_rng(seed)

    (img1, img2), matches, kpts1, kpts2, K1, K2, R, t = pair_matches_data
    failure_result = (img1, img2, 180, 180, 180, 0)

    if len(matches) < 5:
        logger.warning(
            "Pair (%s, %s) failed: only %d matches (need >= 5).", img1, img2, len(matches)
        )
        return failure_result

    try:
        pose = _solve_pair_pose(matches, kpts1, kpts2, K1, K2, th)
    except Exception as e:  # unexpected: surface it, don't hide at debug level
        logger.warning("Pair (%s, %s) raised during pose estimation: %s", img1, img2, e)
        return failure_result

    # estimate_pose returns None for degenerate geometry; handle it explicitly
    # instead of letting the unpacking raise and be swallowed above.
    if pose is None:
        logger.warning("Pair (%s, %s) failed: no valid pose recovered.", img1, img2)
        return failure_result

    R_est, t_est, mask = pose
    T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)
    e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
    e_pose = min(10.0, max(e_t, e_R))
    num_inliers = np.sum(mask)

    logger.debug(
        f"Pair ({img1}, {img2}): e_t={e_t:.2f}, e_R={e_R:.2f}, "
        f"e_pose={e_pose:.2f}, inliers={num_inliers}"
    )
    return (img1, img2, e_t, e_R, e_pose, num_inliers)


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_rng(seed: int = 42) -> None:
    """Set seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # optional: turn off TF32 to avoid tiny nondet diffs on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # If using scaled_dot_product_attention, force the math kernel (deterministic)
    try:
        from torch.backends.cuda import sdp_kernel

        sdp_kernel.enable_flash(False)
        sdp_kernel.enable_mem_efficient(False)
        sdp_kernel.enable_math(True)
    except Exception:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    np.random.seed(seed)
    random.seed(seed)


def parse_pair(pair: str, benchmark_name: str) -> tuple:
    """Parse a line from a .pairs file"""
    if benchmark_name in [
        "megadepth1500",
        "graz4k",
        "megadepth_view",
        "megadepth_air2ground",
        "terrasky3d",
    ]:
        # Pose is stores as "... flat(R) flat(t)""
        parts = pair.strip().split()
        img1, img2 = parts[0], parts[1]
        nums = list(map(float, parts[2:]))

        K1 = np.array(nums[0:9]).reshape(3, 3)
        K2 = np.array(nums[9:18]).reshape(3, 3)
        R = np.array(nums[18:27]).reshape(3, 3)
        t = np.array(nums[27:30]).reshape(3, 1)

    elif benchmark_name in ["scannet1500"]:
        # Pose is stored as a "... flat(P)" with P = [R|t;0 0 0 1]
        parts = pair.strip().split()
        img1, img2 = parts[0], parts[1]
        nums = list(map(float, parts[2:]))
        K1 = np.array(nums[0:9]).reshape(3, 3)
        K2 = np.array(nums[9:18]).reshape(3, 3)
        P = np.array(nums[18:34]).reshape(4, 4)
        R = P[:3, :3]
        t = P[:3, 3:].reshape(3, 1)

    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return img1, img2, K1, K2, R, t


def print_metrics(wrapper: "MethodWrapper", metrics: dict) -> None:
    """Pretty print metrics from a benchmark wrapper"""
    print(f"\nEvaluation results for {wrapper.name}:")
    for k, v in metrics.items():
        if k == "total_pairs":
            print(f"{k:<8}: {v:,}")
        else:
            print(f"{k:<8}: {v:.1f}")


def get_best_device(verbose: bool = False) -> str:
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        if verbose:
            print("Using CPU device")
    return device


def estimate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = 0.999999,
    maxIters: int = 10_000,
) -> Optional[tuple]:
    """Estimate pose using essential matrix.

    Returns ``(R, t, inlier_mask)`` or ``None`` when no valid pose can be
    recovered (fewer than 5 points, or a degenerate essential matrix).
    """
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        cameraMatrix=np.eye(3),
        threshold=norm_thresh,
        prob=conf,
        maxIters=maxIters,
        method=cv2.RANSAC,
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute angle error between two rotation matrices"""
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle error between two vectors"""
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(
    T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray
) -> Tuple[float, float]:
    """Compute pose error given ground truth and estimated pose"""
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors: np.ndarray, thresholds: Sequence[float]) -> List[float]:
    """Compute AUC for pose estimation.

    Note:
        This is the 2D-benchmark pose AUC and is NOT the same metric as
        ``compute_AUC`` in ``benchmarks_3D/utils_benchmark_pose.py``. This version
        returns values in ``[0, 1]`` and uses ``searchsorted`` with the default
        ``side="left"``; they are intentionally distinct.
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapezoid(r, x=e) / t)
    return aucs


def _is_numeric_value(v: object) -> bool:
    """True if ``v`` is a number or a non-empty numeric-looking string."""
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str) and v.strip():
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def _detect_numeric_cols(
    rows: List[Dict[str, object]],
    cols: List[str],
    left_align: set,
    right_align: set,
    fill_missing: str,
) -> set:
    """Columns whose every non-empty value is numeric (skipping forced-align cols)."""
    numeric_cols = set()
    for c in cols:
        if c in left_align or c in right_align:
            continue
        is_numeric = True
        for r in rows:
            v = r.get(c, fill_missing)
            if v == fill_missing or v is None or (isinstance(v, str) and v == ""):
                continue
            if not _is_numeric_value(v):
                is_numeric = False
                break
        if is_numeric:
            numeric_cols.add(c)
    return numeric_cols


def _format_cell(v: object, float_format: str) -> str:
    """Format floats with ``float_format``; everything else via ``str()``."""
    if isinstance(v, float):
        return format(v, float_format)
    return str(v)


def _column_widths(
    rows: List[Dict[str, object]],
    cols: List[str],
    min_widths: Mapping[str, int],
    float_format: str,
    fill_missing: str,
) -> Dict[str, int]:
    """Width per column from header, ``min_widths``, and unclipped formatted cells."""
    widths = {c: max(len(str(c)), min_widths.get(c, 0)) for c in cols}
    for r in rows:
        for c in cols:
            cell = _format_cell(r.get(c, fill_missing), float_format)
            widths[c] = max(widths[c], len(cell))
    return widths


def _clip(s: str, maxw: Optional[int]) -> str:
    """Clip ``s`` to ``maxw`` chars with a trailing ellipsis (no-op if ``maxw`` falsy)."""
    if maxw is None or maxw <= 0 or len(s) <= maxw:
        return s
    return s[: max(0, maxw - 1)] + "…"


def _align(
    s: str,
    col: str,
    width: int,
    left_align: set,
    right_align: set,
    numeric_cols: set,
) -> str:
    """Right-align forced/numeric columns, left-align everything else."""
    if col in right_align:
        return s.rjust(width)
    if col in left_align or col not in numeric_cols:
        return s.ljust(width)
    return s.rjust(width)


def print_table(
    rows: List[Dict[str, object]],
    *,
    columns: Optional[Iterable[str]] = None,
    left_align: Optional[Iterable[str]] = None,
    right_align: Optional[Iterable[str]] = None,
    min_widths: Optional[Mapping[str, int]] = None,
    float_format: str = ".3f",
    fill_missing: str = "",
    clip_widths: Optional[Mapping[str, int]] = None,
    header: bool = True,
    sep: str = " | ",
) -> None:
    """
    Pretty-print a table from a list of dicts (assumes all dicts share the same keys).

    Args:
        rows: List of row dicts.
        columns: Optional explicit column order. Defaults to keys() of the first row.
        left_align: Columns to force left-align (others may be auto-detected).
        right_align: Columns to force right-align.
        min_widths: Dict of minimum widths per column (e.g., {"Method": 18, "Desc": 12}).
        float_format: Format spec for floats (e.g., ".2f" → 2 decimals).
        fill_missing: String to use when a key is missing in a row.
        clip_widths: Dict of maximum widths; values longer than the limit are clipped with “…” .
        header: Whether to print a header row and separator.
        sep: Column separator string.
    """
    if not rows:
        print("(empty)")
        return

    cols = list(columns) if columns is not None else list(rows[0].keys())
    left_align = set(left_align or ())
    right_align = set(right_align or ())
    min_widths = dict(min_widths or {})
    clip_widths = dict(clip_widths or {})

    numeric_cols = _detect_numeric_cols(
        rows, cols, left_align, right_align, fill_missing
    )
    widths = _column_widths(rows, cols, min_widths, float_format, fill_missing)

    def render(value: object, col: str) -> str:
        s = _clip(_format_cell(value, float_format), clip_widths.get(col))
        return _align(s, col, widths[col], left_align, right_align, numeric_cols)

    if header:
        print(sep.join(render(c, c) for c in cols))
        print(sep.join("-" * widths[c] for c in cols))

    for r in rows:
        print(sep.join(render(r.get(c, fill_missing), c) for c in cols))
