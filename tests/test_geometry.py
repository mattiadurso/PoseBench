"""Tests for common.geometry (the de-duplicated geometry helpers)."""

import math

import numpy as np
import torch

from common import geometry


def test_is_torch():
    assert geometry.is_torch(torch.zeros(3)) is True
    assert geometry.is_torch(np.zeros(3)) is False
    assert geometry.is_torch([1, 2, 3]) is False
    assert geometry.is_torch(5) is False


def test_normalize_pixel_coordinates_corners_and_center():
    # shape (H, W) = (4, 2); convention: top-left pixel center at (0.5, 0.5).
    shape = (4, 2)
    xy = torch.tensor([[[0.0, 0.0], [2.0, 4.0], [1.0, 2.0]]])  # 1x3x2 (x, y)
    out = geometry.normalize_pixel_coordinates(xy, shape)
    # x: 2*x/W - 1 ; y: 2*y/H - 1
    expected = torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])
    assert torch.allclose(out, expected)


def test_normalize_pixel_coordinates_does_not_mutate_input():
    xy = torch.tensor([[[1.0, 2.0]]])
    xy_copy = xy.clone()
    geometry.normalize_pixel_coordinates(xy, (4, 2))
    assert torch.equal(xy, xy_copy)


def test_grid_sample_nan_constant_image_returns_constant():
    # A constant image must sample to that constant everywhere inside it.
    img = torch.full((1, 3, 8, 8), 0.5)
    xy = torch.tensor([[[2.0, 2.0], [5.0, 6.0]]])  # 1x2x2, inside the image
    sampled, mask = geometry.grid_sample_nan(xy, img)
    assert sampled.shape == (1, 3, 2)
    assert torch.allclose(sampled, torch.full((1, 3, 2), 0.5))
    assert not mask.any()


def test_grid_sample_nan_input_nan_propagates():
    img = torch.rand(1, 1, 8, 8)
    xy = torch.tensor([[[3.0, 3.0], [float("nan"), float("nan")]]])
    sampled, mask = geometry.grid_sample_nan(xy, img)
    assert torch.isnan(sampled[0, 0, 1])  # nan input -> nan output
    assert not torch.isnan(sampled[0, 0, 0])  # valid input -> valid output
    # The nan-input point is reported as False in the image-nan mask.
    assert mask[0, 1].item() is False or mask[0, 1].item() == 0


def test_grid_sample_nan_out_of_bounds_becomes_nan():
    img = torch.rand(1, 1, 8, 8)
    xy = torch.tensor([[[100.0, 100.0]]])  # far outside
    sampled, _ = geometry.grid_sample_nan(xy, img)
    assert torch.isnan(sampled).all()


def test_grid_sample_nan_handles_3d_image():
    # 3D image (B,H,W) -> result has no channel dimension.
    img = torch.full((1, 8, 8), 0.25)
    xy = torch.tensor([[[2.0, 2.0]]])
    sampled, _ = geometry.grid_sample_nan(xy, img)
    assert sampled.shape == (1, 1)
    assert torch.allclose(sampled, torch.tensor([[0.25]]))


def test_filter_outside_marks_outside_points_nan():
    shape = (10, 10)  # H, W
    xy = torch.tensor([[[5.0, 5.0], [-1.0, 5.0], [5.0, 11.0]]])
    out = geometry.filter_outside(xy, shape)
    assert torch.equal(out[0, 0], torch.tensor([5.0, 5.0]))  # inside, kept
    assert torch.isnan(out[0, 1]).all()  # x < 0
    assert torch.isnan(out[0, 2]).all()  # y >= H


def test_filter_outside_border():
    shape = (10, 10)
    xy = torch.tensor([[[1.0, 1.0]]])
    # With border=2, (1,1) is within the border band -> filtered out.
    out = geometry.filter_outside(xy, shape, border=2)
    assert torch.isnan(out).all()


def test_compute_relative_pose_identity():
    R = torch.eye(3)
    t = torch.zeros(3)
    rots, trans = geometry.compute_relative_pose(R, t, R, t)
    assert torch.allclose(rots, torch.eye(3))
    assert torch.allclose(trans, torch.zeros(3))


def test_compute_relative_pose_known_rotation():
    # 90deg about z for pose 2, identity for pose 1 -> relative is the 90deg rot.
    theta = math.pi / 2
    Rz = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R1, t1 = torch.eye(3), torch.zeros(3)
    rots, trans = geometry.compute_relative_pose(R1, t1, Rz, torch.zeros(3))
    assert torch.allclose(rots, Rz, atol=1e-6)
    assert torch.allclose(trans, torch.zeros(3), atol=1e-6)
