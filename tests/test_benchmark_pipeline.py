"""Tests for the 2D-benchmark parsing and matching/pose pipeline.

These run CPU-only on synthetic inputs. The ``Benchmark`` methods are exercised
on bare instances (``__new__`` + the few attributes each method reads) so they
can be tested in isolation without dataset files on disk.
"""

import logging

import numpy as np
import torch

from benchmarks_2D.utils_benchmark import parse_md1500_poses, parse_pair
from benchmarks_2D.benchmark_parallel import Benchmark


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def test_parse_pair_megadepth_format():
    K = [500, 0, 320, 0, 500, 240, 0, 0, 1]
    R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    t = [0.1, 0.2, 0.3]
    line = "a.jpg b.jpg " + " ".join(str(v) for v in (K + K + R + t))

    img1, img2, K1, K2, R_out, t_out = parse_pair(line, benchmark_name="megadepth1500")

    assert (img1, img2) == ("a.jpg", "b.jpg")
    assert K1.shape == (3, 3) and K2.shape == (3, 3)
    assert K1[0, 0] == 500 and K1[1, 2] == 240
    assert R_out.shape == (3, 3)
    assert t_out.shape == (3, 1)


def test_parse_pair_unknown_benchmark_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_pair("a b 1 2 3", benchmark_name="nope")


def test_parse_md1500_poses(tmp_path):
    poses = tmp_path / "poses.txt"
    poses.write_text(
        "# header line is skipped\n"
        "scene/img1.jpg 1 0 0 0 1 0 0 0 1 0.1 0.2 0.3 PINHOLE 640 480 "
        "500 500 320 240\n"
    )

    views = parse_md1500_poses(poses)

    assert "scene/img1.jpg" in views
    v = views["scene/img1.jpg"]
    assert v["K"][0, 0] == 500 and v["K"][1, 1] == 500
    assert v["K"][0, 2] == 320 and v["K"][1, 2] == 240
    assert v["R"].shape == (3, 3)
    assert v["t"].shape == (3, 1)
    assert v["P"].shape == (4, 4)
    assert v["camera_model"] == "PINHOLE"
    assert list(v["image_size"]) == [640, 480]


# --------------------------------------------------------------------------- #
# Matching pipeline (_match_pairs)
# --------------------------------------------------------------------------- #
def _bare_matcher_benchmark():
    b = Benchmark.__new__(Benchmark)
    b.matcher_params = {"min_score": -1.0}
    b.device = "cpu"
    b.scaling_factor = 1
    b.oom_safe = False
    return b


def _eye_intrinsics():
    K = np.eye(3)
    return K.copy(), K.copy(), np.eye(3), np.zeros((3, 1))


def test_match_pairs_produces_expected_record():
    b = _bare_matcher_benchmark()
    # Identical descriptors -> every keypoint mutually matches its counterpart.
    des = torch.eye(3)
    kpts = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    keypoints_dict = {"a": {"kpts": kpts}, "b": {"kpts": kpts}}
    descriptors_dict = {"a": des, "b": des}
    K1, K2, R, t = _eye_intrinsics()
    pair_data = [(("a", "b"), K1, K2, R, t)]

    out = b._match_pairs(pair_data, keypoints_dict, descriptors_dict)

    assert ("a", "b") in out
    rec = out[("a", "b")]
    assert set(rec) == {"matches", "kpts1", "kpts2", "K1", "K2", "R", "t"}
    assert rec["matches"].shape[1] == 2
    assert rec["matches"].shape[0] == 3  # identity descriptors -> 3 matches
    assert isinstance(rec["kpts1"], np.ndarray)


def test_match_pairs_skips_dropped_image_instead_of_crashing(caplog):
    b = _bare_matcher_benchmark()
    des = torch.eye(2)
    kpts = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    # 'b' was dropped during extraction -> absent from both dicts.
    keypoints_dict = {"a": {"kpts": kpts}}
    descriptors_dict = {"a": des}
    K1, K2, R, t = _eye_intrinsics()
    pair_data = [(("a", "b"), K1, K2, R, t)]

    with caplog.at_level(logging.WARNING):
        out = b._match_pairs(pair_data, keypoints_dict, descriptors_dict)

    assert out == {}  # no crash; the unsatisfiable pair is skipped
    assert any("Skipped" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Pose-estimation path (batch_pose_estimation)
# --------------------------------------------------------------------------- #
def test_batch_pose_estimation_one_result_per_pair_and_restores_env(monkeypatch):
    import os

    b = Benchmark.__new__(Benchmark)
    b.njobs = 1
    b.ransac_th = 0.5
    b.seed = 0

    K = np.eye(3)
    matches_dict = {
        ("a", "b"): {
            "matches": torch.zeros(3, 2, dtype=torch.long),  # < 5 -> failure sentinel
            "kpts1": np.zeros((3, 2)),
            "kpts2": np.zeros((3, 2)),
            "K1": K.copy(),
            "K2": K.copy(),
            "R": np.eye(3),
            "t": np.zeros((3, 1)),
        }
    }

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    results = b.batch_pose_estimation(matches_dict)

    assert len(results) == 1
    assert results[0] == ("a", "b", 180, 180, 180, 0)
    # The temporary CUDA_VISIBLE_DEVICES="" override must be reverted.
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
