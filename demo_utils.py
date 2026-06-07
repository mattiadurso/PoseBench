"""Geometry and plotting helpers for the demo notebook (matches, epipolar lines)."""

import time
from copy import deepcopy
from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from common.geometry import is_torch
from common.geometry import compute_relative_pose  # noqa: F401  re-exported for notebooks


TensorOrArray = Union[torch.Tensor, np.ndarray]


def unproject_points2d(points, K, remove_last=True):
    """
    Unproject 2D points to 3D points.
    """
    points = to_torch(points, b=False)
    K = to_torch(K, b=False)

    points = to_homogeneous(points)
    points_unprojected = (K.inverse() @ points.permute(-1, -2)).permute(-1, -2)

    if remove_last:
        points_unprojected = points_unprojected[:, :2] / points_unprojected[:, 2:]
        return points_unprojected.reshape(-1, 2)

    points_unprojected = points_unprojected / points_unprojected[:, 2:]
    return points_unprojected.reshape(-1, 3)


def to_homogeneous(vector):
    """
    Convert a 2D vector to homogeneous coordinates.
    """
    vector = to_torch(vector, b=False)
    if vector.shape[1] == 2:
        vector = torch.hstack([vector, torch.ones_like(vector)[..., :1]])
    return vector.float()


def compute_epipolar_lines_coeff(
    E: TensorOrArray,  # bx3x3
    points: TensorOrArray,  # bxNx2
    K=None,  # bx3x3
):
    """
    Compute the epipolar lines coefficients from the essential/fundamental
    matrix and the points. It is needed to unproject points if using the
    Essential matrix.
    Args:
        E: essential matrix
        points: points in the image
        K: intrinsics matrix. If not provide E is assumed to be the
           fundamental matrix.
    Returns:
        epi_lines: epipolar lines coefficients
    """
    points = to_torch(points, b=False)
    E = to_torch(E, b=False)[0]
    if K is not None:
        K = to_torch(K, b=False)

    if K is not None:
        points = unproject_points2d(points, K, remove_last=False)
    else:
        points = to_homogeneous(points)

    return (E @ points.T).T  # epipolar coefficients [a,b,c] for each point


def distance_line_points_parallel(line, points):
    """
    line: tensor [1,3], [3], [3,1]
    points: tensor [N,2]
    """
    a, b, c = line.flatten()
    x, y = points[:, 0], points[:, 1]
    return torch.abs(a * x + b * y + c) / (a**2 + b**2) ** 0.5


def to_torch(vector_, b=True):
    """
    Convert a numpy array to a torch tensor. Eventually add batch size.
    """
    vector = deepcopy(vector_)
    if not is_torch(vector):
        vector = torch.tensor(vector)

    if b and len(vector.shape) < 3:
        vector = vector.unsqueeze(0)

    return vector.float()


def compute_fundamental_from_relative_motion(R, t, K0, K1):
    """
    Compute the fundamental matrix from the relative rotation and translation.
    Args:
        R: relative rotation matrix
        t: relative translation vector
        K0: intrinsics matrix of image 0
        K1: intrinsics matrix of image 1
    Returns:
        Fm: fundamental matrix
    """
    R, t = to_torch(R), to_torch(t, b=False)
    K0, K1 = to_torch(K0, b=True), to_torch(K1, b=True)
    Em = compute_essential_from_relative_motion(R, t)
    Fm = torch.bmm(K1.permute(0, 2, 1).inverse(), torch.bmm(Em, K0.inverse()))
    return Fm


def compute_essential_from_relative_motion(R, t):
    """
    Compute the essential matrix from the relative rotation and translation.
    Args:
        R: relative rotation matrix
        t: relative translation vector
    Returns:
        Em: essential matrix
    """
    R = to_torch(R)
    t = to_torch(t, b=False)

    Tx = vector_to_skew_symmetric_matrix(t)
    Em = Tx @ R

    return Em


def vector_to_skew_symmetric_matrix(t):
    """
    Convert a 3D vector to its skew-symmetric matrix.
    Args:
        t: torch.Tensor of shape (3,) or (1, 3)
    Returns:
        Tx: torch.Tensor of shape (3, 3)
    """
    t = t.flatten()
    Tx = torch.zeros(3, 3, dtype=t.dtype, device=t.device)
    Tx[0, 1] = -t[2]
    Tx[0, 2] = t[1]
    Tx[1, 0] = t[2]
    Tx[1, 2] = -t[0]
    Tx[2, 0] = -t[1]
    Tx[2, 1] = t[0]
    return Tx


def plot_imgs(images, titles=None, rows=1):
    """
    Plot images in a grid with the specified number of rows.

    Args:
        images (list of torch.Tensor or numpy.ndarray): List of images to plot.
        titles (list of str, optional): List of titles for each image.
        rows (int, optional): Number of rows in the grid. Default is 1.
    """
    # Calculate number of columns based on the number of rows
    cols = -(-len(images) // rows)  # Ceiling division to handle uneven grids

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 5 * rows))

    # Flatten axes for easy iteration, in case rows or cols == 1
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]

            # Check if image is in C, H, W format and convert to H, W, C if needed
            if isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[0] in [1, 3]:  # C, H, W
                    img = img.permute(1, 2, 0).cpu().numpy()  # Channels first to last

            # Determine colormap based on number of channels
            if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
                cmap = "gray"
            else:
                cmap = None

            ax.imshow(img.squeeze(), cmap=cmap)
            ax.axis("off")

            # Set title if provided
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])
        else:
            ax.axis("off")  # Hide unused axes

    plt.tight_layout()
    plt.show()


def _pad_to_height(img, target_h, pad_color):
    """Vertically center-pad ``img`` to ``target_h``; return (padded, top_offset)."""
    h, w, c = img.shape
    if h >= target_h:
        return img, 0
    pad_top = (target_h - h) // 2
    color_tensor = torch.tensor(pad_color, dtype=img.dtype, device=img.device).view(
        1, 1, 3
    )
    padded = color_tensor.expand(target_h, w, 3).clone()
    padded[pad_top : pad_top + h] = img
    return padded, pad_top


def _draw_match_lines(kpt1, kpt2, offset_x, highlight_bad_matches, F_gt, reth):
    """Draw match lines between the two image panels onto the current axes."""
    if highlight_bad_matches is None:
        for i in range(kpt1.shape[0]):
            plt.plot(
                [kpt1[i, 0], kpt2[i, 0] + offset_x],
                [kpt1[i, 1], kpt2[i, 1]],
                c="b",
                linewidth=0.25,
                alpha=0.75,
            )
        return

    points1 = to_torch(kpt1, b=False)
    points2 = to_torch(kpt2, b=False)
    E12 = to_torch(F_gt)
    E21 = E12.permute(0, 2, 1)

    epilines_A = compute_epipolar_lines_coeff(E12, points1)
    epilines_B = compute_epipolar_lines_coeff(E21, points2)

    repr_err_A = [
        distance_line_points_parallel(epilines_A[i], points2[i][None]).item()
        for i in range(epilines_A.shape[0])
    ]
    repr_err_B = [
        distance_line_points_parallel(epilines_B[i], points1[i][None]).item()
        for i in range(epilines_B.shape[0])
    ]

    reth = 5
    good_matches = (torch.tensor(repr_err_A) <= reth) & (torch.tensor(repr_err_B) <= reth)

    kpts1_good, kpts2_good = kpt1[good_matches], kpt2[good_matches]
    kpts1_bad, kpts2_bad = kpt1[~good_matches], kpt2[~good_matches]
    print(f"Reprojection error threshold: {reth} pixels")
    print(f"Inliers: {kpts1_good.shape[0]}/{kpt1.shape[0]}")

    for k1, k2, color in (
        (kpts1_good, kpts2_good, "g"),
        (kpts1_bad, kpts2_bad, "r"),
    ):
        for i in range(k1.shape[0]):
            plt.plot(
                [k1[i, 0], k2[i, 0] + offset_x],
                [k1[i, 1], k2[i, 1]],
                c=color,
                linewidth=3,
                alpha=0.85,
            )


def _pad_to_common_canvas(img1, img2, c):
    """Pad the smaller image with white so both share the larger canvas size."""
    if img1.shape[0] < img2.shape[0]:
        img1 = torch.cat(
            (
                img1,
                torch.ones((img2.shape[0] - img1.shape[0], img1.shape[1], c)).int()
                * 255,
            ),
            dim=0,
        )
    else:
        img2 = torch.cat(
            (
                img2,
                torch.ones((img1.shape[0] - img2.shape[0], img2.shape[1], c)).int()
                * 255,
            ),
            dim=0,
        )

    if img1.shape[1] < img2.shape[1]:
        img1 = torch.cat(
            (
                img1,
                torch.ones((img1.shape[0], img2.shape[1] - img1.shape[1], c)).int()
                * 255,
            ),
            dim=1,
        )
    else:
        img2 = torch.cat(
            (
                img2,
                torch.ones((img2.shape[0], img1.shape[1] - img2.shape[1], c)).int()
                * 255,
            ),
            dim=1,
        )
    return img1, img2


def _scatter_kpts(kpt1, kpt2, offset_x, index, sample_points):
    """Scatter (optionally subsampled, optionally numbered) keypoints on both panels.

    Returns:
        The (possibly subsampled) ``(kpt1, kpt2)`` so later drawing stays in sync.
    """
    if sample_points and sample_points < len(kpt1):
        kpt1 = kpt1[:: len(kpt1) // sample_points]
        kpt2 = kpt2[:: len(kpt2) // sample_points]

    if index:
        for i, (x, y) in enumerate(kpt1):
            plt.text(x, y, c="w", s=str(i), fontsize=6, ha="center", va="center")
        for i, (x, y) in enumerate(kpt2):
            plt.text(
                x + offset_x,
                y,
                c="w",
                s=str(i),
                fontsize=6,
                ha="center",
                va="center",
            )

    plt.scatter(kpt1[:, 0], kpt1[:, 1], c="r", s=2)
    plt.scatter(kpt2[:, 0] + offset_x, kpt2[:, 1], c="r", s=2)
    return kpt1, kpt2


def _finalize_plot(text, text_font, axis, plot_name):
    """Add an optional caption, hide axes, optionally save, then show the figure."""
    if text is not None:
        # plot text just above the image, align text to the top left corner
        plt.text(
            0.5,
            1.01,
            text,
            transform=plt.gca().transAxes,
            fontsize=text_font,
            ha="center",
            va="bottom",
        )
    plt.axis("off" if axis else "on")
    plt.tight_layout()
    if plot_name is not None:
        time_ = time.strftime("%Y-%m-%d_%H-%M-%S")
        plot_name = plot_name if plot_name is not None else f"plot_{time_}"
        plt.savefig(plot_name + ".png", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()


def plot_imgs_and_kpts(
    img1,
    img2,
    kpt1,
    kpt2,
    space=100,
    matches=True,
    index=False,
    sample_points=32,
    pad=False,
    figsize=(10, 5),
    axis=True,
    scatter=True,
    highlight_bad_matches=None,
    F_gt=None,
    plot_name=None,
    reth=5,
    text=None,
    text_font=12,
):
    """
    Plot two images side by side with keypoints overlayed and matches if specified.
    """
    assert img1.shape[-1] == img2.shape[-1], "Images must have the same channels"
    assert img1.shape[-1] in [1, 3], "Images must be RGB"
    c = img1.shape[-1]
    # check if images are numpy, then to tensor
    if isinstance(img1, np.ndarray):
        img1 = torch.tensor(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.tensor(img2)

    # Determine target height and pad both images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    target_h = max(h1, h2)

    img1, offset1 = _pad_to_height(img1, target_h, (255, 255, 255))
    img2, offset2 = _pad_to_height(img2, target_h, (255, 255, 255))

    # Adjust keypoints for vertical padding (y coordinate)
    kpt1 = deepcopy(kpt1).astype(np.float32)
    kpt2 = deepcopy(kpt2).astype(np.float32)
    kpt1[:, 1] += offset1
    kpt2[:, 1] += offset2

    if pad:
        # find smaller image and put it in a white canvas of the bigger image size
        img1, img2 = _pad_to_common_canvas(img1, img2, c)

    white = torch.ones((img1.shape[0], space, c)).int() * 255
    concat = torch.cat((img1, white, img2), dim=1).int()

    plt.figure(figsize=figsize)
    plt.imshow(concat)

    offset_x = img1.shape[1] + space
    if scatter:
        kpt1, kpt2 = _scatter_kpts(kpt1, kpt2, offset_x, index, sample_points)

    if matches:
        _draw_match_lines(kpt1, kpt2, offset_x, highlight_bad_matches, F_gt, reth)

    _finalize_plot(text, text_font, axis, plot_name)
