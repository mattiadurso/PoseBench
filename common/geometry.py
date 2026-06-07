"""Geometry and tensor-sampling helpers shared across PoseBench.

These functions were previously duplicated in ``wrappers/wrapper.py`` and the
``benchmarks_2D``/``benchmarks_3D`` utility modules. They are collected here as the
single source of truth; the original locations now delegate to these definitions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def is_torch(vector: object) -> bool:
    """Check whether a value is a torch tensor.

    Args:
        vector: Any object.

    Returns:
        True if ``vector`` is a ``torch.Tensor``, False otherwise.
    """
    return isinstance(vector, torch.Tensor)


def compute_relative_pose(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the relative pose between two camera poses.

    Args:
        R1: Rotation of the first pose (3x3).
        t1: Translation of the first pose (3,).
        R2: Rotation of the second pose (3x3).
        t2: Translation of the second pose (3,).

    Returns:
        A tuple ``(rots, trans)`` with the relative rotation and translation that
        map the first pose into the second.
    """
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def normalize_pixel_coordinates(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray
) -> Tensor:
    """Normalize pixel coordinates from ``[0, W/H]`` to ``[-1, 1]``.

    The corner ``(-1, -1)`` is the exact top-left corner of the image, so the input
    must follow the convention where the center of the top-left pixel is at
    ``(0.5, 0.5)``. ``xy`` is ordered as ``(x, y)`` and ``shape`` as ``(H, W)``.

    Args:
        xy: Input coordinates in ``(x, y)`` order, shape ``...x2``.
        shape: Image shape in ``(H, W)`` order.

    Returns:
        Coordinates normalized to ``[-1, 1]``, same shape as ``xy``.
    """
    xy_norm = xy.clone()
    # The shape indices are flipped because the coordinates are (x, y) but shape is (H, W).
    xy_norm[..., 0] = 2 * xy_norm[..., 0] / shape[1]
    xy_norm[..., 1] = 2 * xy_norm[..., 1] / shape[0]
    xy_norm -= 1
    return xy_norm


def grid_sample_nan(
    xy: Tensor, img: Tensor, mode: str = "nearest"
) -> tuple[Tensor, Tensor]:
    """``grid_sample`` with coordinate normalization and NaN handling.

    If a NaN is present in ``xy`` the corresponding output is NaN. Works with input
    of shape ``B,n,2`` and ``B,n0,n1,2``. Points that fall outside the image are
    treated as NaN (points very close to the border are interpolated with the
    ``border`` padding mode).

    Args:
        xy: Input coordinates (top-left pixel center at ``(0.5, 0.5)``),
            shape ``B,n,2`` or ``B,n0,n1,2``.
        img: Image to sample from, shape ``BxCxHxW`` or ``BxHxW``.
        mode: Interpolation mode.

    Returns:
        A tuple ``(sampled, mask_img_nan)``:
            sampled: Sampled values, ``BxCxN`` or ``BxCxN0xN1`` (without the channel
                dimension if the input image had none).
            mask_img_nan: Mask of points whose sampled image value was NaN. Points
                in ``xy`` that were themselves NaN appear as False, to distinguish
                invalid sampling positions from valid positions with a NaN value.
    """
    assert img.dim() in {3, 4}
    if img.dim() == 3:
        # Remove the channel dimension from the result at the end of the function.
        squeeze_result = True
        img.unsqueeze_(1)
    else:
        squeeze_result = False

    assert xy.shape[-1] == 2
    assert xy.dim() == 3 or xy.dim() == 4
    B, C, H, W = img.shape

    xy_norm = normalize_pixel_coordinates(xy, img.shape[-2:])  # BxNx2 or BxN0xN1x2
    # Set to NaN the points that fall out of the image.
    xy_norm[(xy_norm < -1) + (xy_norm > 1)] = float("nan")
    if xy.ndim == 3:
        sampled = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=False,
            mode=mode,
            padding_mode="border",
        ).view(B, C, xy.shape[1])  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=False, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # Points that are not NaN but sampled a NaN image value. The sum squashes channels.
    mask_img_nan = torch.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # Set sampled values to NaN for points that were NaN (grid_sample treats them as (-1, -1)).
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    if xy.ndim == 3:
        sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    else:
        sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img.squeeze_(1)
        sampled.squeeze_(1)

    return sampled, mask_img_nan


def filter_outside(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray, border: int = 0
) -> Tensor:
    """Set to NaN all points outside the rectangle defined by ``shape`` (H, W).

    Args:
        xy: Keypoints with coordinates ``(x, y)``, shape ``(B)xnx2``.
        shape: Shape that should contain the keypoints, ``(H, W)``.
        border: Minimum border to apply when masking.

    Returns:
        Input keypoints with ``nan`` where either coordinate fell outside ``shape``,
        shape ``(B)xnx2``.
    """
    assert xy.shape[-1] == 2, f"xy must have last dimension of size 2, got {xy.shape}"
    assert len(shape) == 2, f"shape must be a tuple of 2 elements, got {shape}"
    assert border < max(shape), (
        f"border must be smaller than the smallest shape dimension, got {border} and {shape}"
    )

    xy = xy.clone()
    outside_mask = (
        (xy[..., 0] < border)
        + (xy[..., 0] >= shape[1] - border)
        + (xy[..., 1] < border)
        + (xy[..., 1] >= shape[0] - border)
    )  # (B)xn
    xy[outside_mask] = float("nan")
    return xy
