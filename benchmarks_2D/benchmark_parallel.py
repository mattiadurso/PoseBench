"""Parallel 2D pose-estimation benchmark driver for sparse/dense feature methods."""

# This code is based on Parskatt implementation in DKM and DeDoDe.

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

abs_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(abs_root))

import warnings

warnings.filterwarnings("ignore")

import gc
import time
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from functools import partial
from joblib import Parallel, delayed

from matchers.mnn import MNN
from benchmarks_2D.utils_benchmark import (
    fix_rng,
    parse_pair,
    print_metrics,
    pose_auc,
    process_pose_estimation,
    parse_poses,
    load_depth,
)
from benchmarks_2D.repeatability_utils import compute_repeatabilities_from_kpts


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


class Benchmark:
    """Runs a 2D matching benchmark: feature extraction, matching, and pose estimation."""

    def __init__(
        self,
        benchmark_name: str,
        dataset_path=abs_root / "benchmarks_2D/megadepth1500/data",
        ransac_th: float = 1,
        min_score: float = 0.5,
        ratio_test: float = 1,
        max_kpts: int = 2048,
        njobs: int = 8,
        seed: int = 0,
        scaling_factor: float = 1.0,
        ghr_partial: bool = False,
        feature_path: str = None,
        reuse_kpts_path: str = None,
        save_descriptors: bool = False,
        compute_repeatability: bool = False,
        device: str = "cuda",
        oom_safe: bool = False,
        px_thrs: list = [1, 3, 5],
    ):
        """Store benchmark configuration and load pairs, paths, and any cached features."""
        self.benchmark_name = benchmark_name
        self.dataset_path = dataset_path
        self.ransac_th = ransac_th
        self.min_score = min_score
        self.ratio_test = ratio_test
        self.max_kpts = max_kpts
        self.njobs = njobs
        self.seed = seed
        self.scaling_factor = scaling_factor  # mostly for GHR
        self.ghr_partial = ghr_partial
        self.oom_safe = oom_safe
        self.feature_path = feature_path
        self.reuse_kpts_path = reuse_kpts_path
        self.save_descriptors = save_descriptors
        if scaling_factor != 1 and compute_repeatability:
            logger.warning(
                "Repeatability computation might be incorrect when "
                + "scaling_factor != 1. Thus, it is disabled."
            )
        self.compute_repeatability = (
            compute_repeatability
            and benchmark_name
            in [
                "megadepth1500",
                "graz4k",
            ]
            and scaling_factor == 1
        )  # repeatability only for MegaDepth and Graz4K
        self.device = device
        self.px_thrs = px_thrs

        s = " with repeatability computation" if self.compute_repeatability else ""
        logger.info(f"Benchmarking {self.benchmark_name}{s}.")

        self._load_calibrated_pairs()
        self._resolve_dataset_paths()
        self.matcher_params = {"min_score": min_score, "ratio_test": ratio_test}
        self._load_precomputed_features()

    def _load_calibrated_pairs(self):
        """Load calibrated pairs, drop the header, and apply scene filters."""
        self.dataset_path = abs_root / self.dataset_path
        with open(self.dataset_path / "pairs_calibrated.txt", "r") as f:
            self.pairs_calibrated = f.read().splitlines()

        # skip header starting with #
        self.pairs_calibrated = [p for p in self.pairs_calibrated if p and p[0] != "#"]

        # exlude some scenes for terrasky testing
        # Only when testing on terrasky3d, whose test split contains these Graz
        # scenes. Applying it unconditionally also stripped them from graz4k
        # itself, which dropped ~2,000 of its 4,411 pairs and made --ghr-partial
        # (which then selects graz_main_square) return zero pairs.
        if self.benchmark_name.lower() == "terrasky3d":
            scenes = ["graz_clocktower", "graz_main_square", "graz_castle"]
            self.pairs_calibrated = [
                p for p in self.pairs_calibrated if not any(s in p for s in scenes)
            ]

        logger.info(
            f"Loaded {len(self.pairs_calibrated):,} calibrated pairs from {self.dataset_path / 'pairs_calibrated.txt'}"
        )

        if self.ghr_partial:
            scene = "graz_main_square"  # small scene for quick testing
            self.pairs_calibrated = [p for p in self.pairs_calibrated if scene in p]
            logger.info(f"Using only pairs from {scene} for GHR partial benchmark.")

    def _resolve_dataset_paths(self):
        """Resolve image/depth/views paths for the configured benchmark."""
        name = self.benchmark_name.lower()
        if name == "megadepth1500":
            self.images_path = self.dataset_path / "images"
            self.depths_path = self.dataset_path / "depths"
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        elif name in ["megadepth_air2ground", "megadepth_view"]:
            self.images_path = self.dataset_path / "images"
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        elif name == "scannet1500":
            self.images_path = self.dataset_path

        elif name == "graz4k":
            self.images_path = self.dataset_path
            self.depths_path = self.dataset_path
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        elif name == "terrasky3d":
            self.images_path = self.dataset_path
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        else:
            raise ValueError(f"Unknown dataset name: {self.benchmark_name}")

    def _load_precomputed_features(self):
        """Load precomputed keypoints/descriptors if a feature path was given."""
        if self.feature_path is not None:
            self.feature_path = Path(self.feature_path)
            self.keypoints_dict = torch.load(
                self.feature_path / "keypoints.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.descriptors_dict = torch.load(
                self.feature_path / "descriptors.pt",
                map_location="cpu",
                weights_only=False,
            )
            logger.info(
                f"Using precomputed features from {self.feature_path / 'keypoints.pt'} and {self.feature_path / 'descriptors.pt'}"
            )
        else:
            self.keypoints_dict = None
            self.descriptors_dict = None
            logger.info("Extracting features using wrapper")

        # Keypoints-only reuse. Unlike feature_path (which also loads
        # descriptors and skips extraction entirely), this keeps extraction but
        # feeds the stored keypoints to the detector as custom_kpts, so only the
        # descriptor stage runs. Intended for a custom-descriptor run that must
        # describe exactly the keypoints a previous baseline run detected --
        # which is what SANDesc does by construction. Keypoint *scores* are not
        # stored and are not used downstream (_match_pairs reads only "kpts"
        # and the descriptors), so discarding them is safe.
        if self.reuse_kpts_path is not None:
            self.reuse_kpts_path = Path(self.reuse_kpts_path)
            self.reuse_kpts_dict = torch.load(
                self.reuse_kpts_path / "keypoints.pt",
                map_location="cpu",
                weights_only=False,
            )
            logger.info(
                f"Reusing keypoints from {self.reuse_kpts_path / 'keypoints.pt'} "
                f"({len(self.reuse_kpts_dict):,} images); descriptors recomputed."
            )
        else:
            self.reuse_kpts_dict = None

    def extract_features_with_wrapper(self, wrapper):
        """Extract features using the wrapper."""

        keypoints_dict = {}
        descriptors_dict = {}

        # Get all unique images
        unique_images = set()
        for pair in self.pairs_calibrated:
            img1, img2, _, _, _, _ = parse_pair(
                pair, benchmark_name=self.benchmark_name
            )
            unique_images.add(img1)
            unique_images.add(img2)

        # Extract features for each image
        s = " and depths" if self.compute_repeatability else ""
        for img_name in tqdm(unique_images, desc=f"Extracting features{s}"):
            img_path = self.images_path / img_name

            try:
                img = wrapper.load_image(img_path, scaling=self.scaling_factor)

                custom_kpts = None
                if self.reuse_kpts_dict is not None:
                    stored = self.reuse_kpts_dict.get(img_name)
                    if stored is None:
                        logger.warning(
                            f"{img_name} missing from reused keypoints, detecting"
                        )
                    else:
                        custom_kpts = stored["kpts"]

                with torch.no_grad():
                    out = wrapper.extract(img, self.max_kpts, custom_kpts=custom_kpts)

                keypoints_dict[img_name] = {"kpts": out.kpts.detach().cpu()}
                descriptors_dict[img_name] = out.des.detach().cpu()

                if self.compute_repeatability:
                    keypoints_dict[img_name]["depth"] = self._sample_keypoint_depth(
                        wrapper, img_name, out.kpts
                    )

            except Exception as e:
                logger.warning(f"Error processing {img_name}: {e}")
                continue

            # This might slow down a bit, but some methods might go OOM. Don't use if not needed
            if self.oom_safe:
                del out, img
                gc.collect()
                torch.cuda.empty_cache()

        return keypoints_dict, descriptors_dict

    def _sample_keypoint_depth(self, wrapper, img_name, kpts):
        """Load the depth map for an image and sample it at the keypoints."""
        if self.benchmark_name == "megadepth1500":
            Z_path = self.depths_path / f"{img_name.split('.')[0]}.h5"
        elif self.benchmark_name == "graz4k":
            scene, _, cam, image_name = img_name.split("/")
            Z_path = (
                self.depths_path
                / scene
                / "depth"
                / cam
                / f"{image_name.split('.')[0]}.h5"
            )

        Z = load_depth(Z_path, scale_factor=self.scaling_factor, target=kpts)
        Z_sampled, _ = wrapper.grid_sample_nan(kpts[None], Z[None], mode="nearest")
        return Z_sampled[0].detach()

    def save_features_to_intermediate(self, keypoints_dict, descriptors_dict, key):
        """Save extracted features to intermediate directory.
        key: f"{wrapper_name}_kpts_{max_kpts}"
        """
        intermediate_path = (
            Path(f"benchmarks_2D/{self.benchmark_name}/intermediate") / key
        )
        os.makedirs(intermediate_path, exist_ok=True)

        keypoints_file = intermediate_path / "keypoints.pt"
        descriptors_file = intermediate_path / "descriptors.pt"

        torch.save(keypoints_dict, keypoints_file)

        # Descriptors are not saved by default: they dwarf everything else and
        # fill the disk. At 30k keypoints one run writes ~34 GB of descriptors
        # against ~0.7 GB of keypoints, so a handful of runs exhausts the drive
        # (this happened: 111 GB of caches, disk at 98%). Keypoints are kept
        # because --reuse-kpts needs only those. Pass --save-descriptors if you
        # specifically need --features, which reloads both and skips extraction.
        if self.save_descriptors:
            torch.save(descriptors_dict, descriptors_file)
            logger.info(f"Features saved: {keypoints_file} and {descriptors_file}")
            return keypoints_file, descriptors_file

        logger.info(f"Keypoints saved: {keypoints_file} (descriptors not saved)")
        return keypoints_file, None

    def batch_match_all_pairs(self, wrapper, save_key=None):
        """Match all pairs in batch mode."""

        # Get features (either precomputed or extract with wrapper)
        if self.feature_path is not None:
            keypoints_dict = self.keypoints_dict
            descriptors_dict = self.descriptors_dict
        else:
            keypoints_dict, descriptors_dict = self.extract_features_with_wrapper(
                wrapper
            )

            # Save features to intermediate directory if key provided
            if save_key:
                self.save_features_to_intermediate(
                    keypoints_dict, descriptors_dict, save_key
                )

        # Prepare all pair data
        pair_data = []
        for pair in self.pairs_calibrated:
            img1, img2, K1, K2, R, t = parse_pair(
                pair, benchmark_name=self.benchmark_name
            )
            pair_data.append(((img1, img2), K1, K2, R, t))

        matches_dict = self._match_pairs(pair_data, keypoints_dict, descriptors_dict)
        rep_results = (
            self._compute_repeatability(pair_data, keypoints_dict)
            if self.compute_repeatability
            else {}
        )
        return matches_dict, rep_results

    def _match_pairs(self, pair_data, keypoints_dict, descriptors_dict):
        """Run the matcher over every prepared pair and collect pose-estimation data."""
        matcher = MNN(**self.matcher_params)
        matches_dict = {}
        skipped = 0
        for (img1, img2), K1, K2, R, t in tqdm(pair_data, desc="Matching pairs"):
            # An image whose feature extraction failed is absent from the dicts;
            # skip the pair instead of crashing on a KeyError downstream.
            if not all(
                img in keypoints_dict and img in descriptors_dict
                for img in (img1, img2)
            ):
                skipped += 1
                continue

            kpts1 = keypoints_dict[img1]["kpts"]
            kpts2 = keypoints_dict[img2]["kpts"]
            desc1 = descriptors_dict[img1].to(self.device)
            desc2 = descriptors_dict[img2].to(self.device)

            # Scale intrinsics if scaling applied
            if self.scaling_factor != 1:
                K1[:2, :3] /= self.scaling_factor
                K2[:2, :3] /= self.scaling_factor

            matches = matcher.match([desc1], [desc2])[0].matches.cpu()
            matches_dict[(img1, img2)] = {
                "matches": matches,
                "kpts1": kpts1.cpu().numpy(),
                "kpts2": kpts2.cpu().numpy(),
                "K1": K1,
                "K2": K2,
                "R": R,
                "t": t,
            }

            if self.oom_safe:
                del desc1, desc2, matches
                gc.collect()
                torch.cuda.empty_cache()

        if skipped:
            logger.warning(
                "Skipped %d/%d pairs with missing features (extraction failed).",
                skipped,
                len(pair_data),
            )
        return matches_dict

    def _pair_repeatability(self, pair, keypoints_dict):
        """Compute the repeatability dict for a single image pair."""
        img1, img2 = pair[:1][0]

        kpts1 = keypoints_dict[img1]["kpts"]
        Z1 = keypoints_dict[img1]["depth"]
        K1 = self.views_dict[img1]["K"]
        P1 = self.views_dict[img1]["P"]
        img1_size = self.views_dict[img1]["image_size"]

        kpts2 = keypoints_dict[img2]["kpts"]
        Z2 = keypoints_dict[img2]["depth"]
        K2 = self.views_dict[img2]["K"]
        P2 = self.views_dict[img2]["P"]
        img2_size = self.views_dict[img2]["image_size"]

        return compute_repeatabilities_from_kpts(
            kpts1[None].float().to(self.device),
            kpts2[None].float().to(self.device),
            K1[None].float().to(self.device),
            K2[None].float().to(self.device),
            Z1[None].float().to(self.device),
            Z2[None].float().to(self.device),
            P1[None].float().to(self.device),
            P2[None].float().to(self.device),
            img1_shape=img1_size,
            img2_shape=img2_size,
            px_thrs=self.px_thrs,
        )

    def _compute_repeatability(self, pair_data, keypoints_dict):
        """Compute repeatability metrics averaged over all pairs."""
        rep_results = {
            **{f"rep_{int(pix)}": [] for pix in self.px_thrs},
            **{f"rep_mnn_{int(pix)}": [] for pix in self.px_thrs},
        }

        # runs in ~5s with 2048kpts, batching it might be even faster
        for pair in tqdm(pair_data, desc="Repeatability"):
            rep = self._pair_repeatability(pair, keypoints_dict)

            for b in rep:
                for k in rep[b]:
                    rep_results[k].append(rep[b][k])

            # clean up
            if self.oom_safe:
                del rep
                gc.collect()
                torch.cuda.empty_cache()

        # average over all pairs
        for k in rep_results:
            rep_results[k] = sum(rep_results[k]) / len(rep_results[k])
        return rep_results

    def batch_pose_estimation(self, matches_dict):
        """Perform pose estimation in parallel."""
        # Prepare data for parallel processing - no batching needed
        all_pairs = []
        for (img1, img2), data in matches_dict.items():
            all_pairs.append(
                (
                    (img1, img2),
                    data["matches"],
                    data["kpts1"],
                    data["kpts2"],
                    data["K1"],
                    data["K2"],
                    data["R"],
                    data["t"],
                )
            )

        # Hide GPUs
        _prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Process pairs in parallel
        pose_estimation_partial = partial(
            process_pose_estimation, th=self.ransac_th, seed=self.seed
        )

        results = Parallel(n_jobs=self.njobs, verbose=0)(
            delayed(pose_estimation_partial)(pair)
            for pair in tqdm(all_pairs, desc="Pose Estimation")
        )

        # Revert GPU settings
        if _prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev

        return results

    def match_with_matcher(self, wrapper):
        """Match all pairs using a matcher wrapper (like RoMa or LightGlue)."""

        matches_dict = {}

        for pair in tqdm(self.pairs_calibrated, desc="Matching pairs with matcher"):
            img1, img2, K1, K2, R, t = parse_pair(
                pair, benchmark_name=self.benchmark_name
            )

            # Match
            matches, kpts1, kpts2 = wrapper.match_pair(
                self.images_path / img1,
                self.images_path / img2,
                max_kpts=self.max_kpts,
            )

            # Scale intrinsics if scaling applied
            if self.scaling_factor != 1:
                K1[:2, :3] /= self.scaling_factor
                K2[:2, :3] /= self.scaling_factor

            # Store for pose estimation
            matches_dict[(img1, img2)] = {
                "matches": matches,
                "kpts1": kpts1.cpu().numpy(),
                "kpts2": kpts2.cpu().numpy(),
                "K1": K1,
                "K2": K2,
                "R": R,
                "t": t,
            }

        return matches_dict

    @torch.no_grad()
    def benchmark(self, wrapper, save_key=None):
        """Run the complete benchmark."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Fixing randomness for parallel processing
        fix_rng(seed=self.seed)

        # Create save key for features with timestamp (similar to GHR pattern)
        features_save_key = f"{save_key}_{timestamp}" if save_key else None

        # Phase 1: Batch matching (with feature extraction if needed)
        wrapper.move_to(self.device)  # ensure wrapper is on the correct device

        if wrapper.is_sparse_feature_extractor:
            matches_dict, rep_results = self.batch_match_all_pairs(
                wrapper, save_key=features_save_key
            )
        else:  # using a matcher like RoMa or LightGlue
            matches_dict = self.match_with_matcher(wrapper)

        wrapper.move_to("cpu")  # to avoid issues in parallel jobs

        # Phase 2: Parallel pose estimation as
        # results = (img1, img2, e_t, e_R, e_pose, inliers)
        results = self.batch_pose_estimation(matches_dict)
        self._save_pair_results(results, save_key)

        wrapper.move_to(self.device)  # move back to original device

        out = self._summarize_pose_results(results)
        if self.compute_repeatability:
            out.update(rep_results)
        self._round_metrics(out)
        return out, timestamp

    def _save_pair_results(self, results, save_key):
        """Write per-pair errors and the unregistered subset to CSV files."""
        base = f"benchmarks_2D/{self.benchmark_name}/results"
        os.makedirs(f"{base}/df_pairs", exist_ok=True)
        with open(f"{base}/df_pairs/{save_key}.csv", "w") as f:
            f.write("img1,img2,e_t,e_R,e_pose,inlier\n")
            for img1, img2, e_t, e_R, e_pose, inlier in results:
                f.write(f"{img1},{img2},{e_t},{e_R},{e_pose},{inlier}\n")

        # optional, save images not registered
        os.makedirs(f"{base}/not_registered", exist_ok=True)
        with open(f"{base}/not_registered/{save_key}.csv", "w") as f:
            f.write("img1,img2,e_t,e_R,e_pose,inlier\n")
            for img1, img2, e_t, e_R, e_pose, inlier in results:
                if e_pose >= 180:
                    f.write(f"{img1},{img2},{e_t},{e_R},{e_pose},{inlier}\n")

    @staticmethod
    def _summarize_pose_results(results):
        """Aggregate per-pair results into AUC / inlier / registration metrics."""
        tot_e_pose = np.array([r[4] for r in results])
        inliers = np.array([r[5] for r in results])
        auc = pose_auc(tot_e_pose, [5, 10, 20])
        unregistred = [r for r in results if r[4] >= 180]
        if unregistred:
            logger.warning(
                "%d/%d pairs failed pose estimation (unregistered).",
                len(unregistred),
                len(results),
            )
        return {
            "inliers": np.mean(inliers),
            "unregistered_pairs": int(len(unregistred)),
            "total_pairs": int(len(results)),
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
        }

    @staticmethod
    def _round_metrics(out):
        """Round metrics in place: counts to 1 decimal, rates scaled to percent."""
        for k in out:
            out[k] = (
                round(out[k], 1)
                if k in ("inliers", "unregistered_pairs", "total_pairs")
                else round(out[k] * 100, 1)
            )


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds", type=str, default="terrasky3d", help="Dataset (ds) name"
    )
    parser.add_argument("--device", default="cuda", help="Device to use for matching")
    parser.add_argument(
        "--wrapper-name", type=str, default="disk-kornia", help="Wrapper name"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset",
    )
    parser.add_argument(
        "--njobs", type=int, default=-1, help="Number of parallel jobs"
    )  # might slightly affect reproducibility and results
    parser.add_argument(
        "--ratio-test", type=float, default=1.0, help="Ratio test threshold"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0, help="Minimum match score"
    )
    parser.add_argument(
        "--ransac-th", type=float, default=1.0, help="Pose estimation threshold"
    )
    parser.add_argument(
        "--max-kpts",
        type=int,
        choices=[2048, 4096, 8000, 30000],
        default=2048,
        help="Maximum keypoints (allowed values: 2048, 4096, 8000, or 30000)",
    )
    parser.add_argument("--run-tag", type=str, default=None, help="Tag for this run")
    parser.add_argument(
        "--custom-desc", type=str, default=None, help="Path to custom descriptors"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Down-scaling factor for input images (e.g., 2 for half size)",
    )
    parser.add_argument(
        "--ghr-partial",
        action="store_true",
        help="Compute partial GHR benchmark (only graz_main_square scene)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to precomputed features (optional)",
    )

    parser.add_argument(
        "--save-descriptors",
        action="store_true",
        help=(
            "Also cache descriptors.pt alongside keypoints.pt. Off by default: "
            "at 30k keypoints this is ~34 GB per run. Only needed for --features."
        ),
    )

    parser.add_argument(
        "--reuse-kpts",
        default=None,
        help=(
            "Path to a previous run's intermediate dir; reuse its keypoints.pt "
            "and recompute descriptors. Unlike --features (which also reuses "
            "descriptors and skips extraction), this only skips detection -- "
            "use it for a --custom-desc run that must describe exactly the "
            "keypoints a baseline run detected."
        ),
    )

    parser.add_argument(
        "--skip-repeatability", action="store_false", help="Don't compute repeatability"
    )
    parser.add_argument(
        "--oom-safe",
        action="store_true",
        help="Use OOM-safe feature extraction (might be slower) when method goes OOM.",
    )

    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    benchmark_name = args.ds

    if benchmark_name.lower() in ["md", "md1500", "megadepth1500"]:
        benchmark_name = "megadepth1500"
    elif benchmark_name.lower() in ["mdv", "megadepth_view"]:
        benchmark_name = "megadepth_view"
    elif benchmark_name.lower() in ["mda", "mda2g", "md_air2ground"]:
        benchmark_name = "megadepth_air2ground"
    elif benchmark_name.lower() in ["sc", "scannet", "sc1500", "scannet1500"]:
        benchmark_name = "scannet1500"
    elif benchmark_name.lower() in ["graz", "g4k", "graz4k"]:
        benchmark_name = "graz4k"
    elif benchmark_name.lower() in ["terrasky3d", "ts3d"]:
        benchmark_name = "terrasky3d"
    else:
        raise ValueError(f"Unknown dataset name: {benchmark_name}")

    data_path = (
        args.data_path
        if args.data_path is not None
        else f"benchmarks_2D/{benchmark_name}/data"
    )
    njobs = args.njobs if args.njobs != -1 else os.cpu_count()
    njobs = 1 if sys.gettrace() else njobs
    ratio_test = args.ratio_test
    min_score = args.min_score
    ransac_th = args.ransac_th
    max_kpts = args.max_kpts
    run_tag = args.run_tag
    custom_desc = args.custom_desc
    seed = args.seed
    scaling_factor = args.scaling_factor
    ghr_partial = args.ghr_partial
    oom_safe = args.oom_safe
    feature_path = args.features
    compute_repeatability = args.skip_repeatability

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    if custom_desc is not None:
        #  Eventually add my descriptors
        weights = torch.load(custom_desc, weights_only=False)
        config = weights["config"]["model"]
        model = {
            "ch_in": config["unet_ch_in"],
            "kernel_size": config["unet_kernel_size"],
            "activ": config["unet_activ"],
            "norm": config["unet_norm"],
            "skip_connection": config["unet_with_skip_connections"],
            "spatial_attention": config["unet_spatial_attention"],
            "third_block": config["third_block"],
        }

        from sandesc_models.sandesc.network_descriptor import SANDesc

        network = SANDesc(**model).eval()

        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights["state_dict"])

        wrapper.add_custom_descriptor(network)
        wrapper.name = f"{wrapper.name}+SANDesc"
        logger.info(f"Using custom descriptors from {custom_desc}.")

    # matcher params
    if benchmark_name == "graz4k" and ghr_partial:
        key = f"{wrapper.name} min_score_{min_score}_ratio_test_{ratio_test}_th_{ransac_th}_mnn_scale_{scaling_factor}_partial {max_kpts}"
    else:
        key = f"{wrapper.name} min_score_{min_score}_ratio_test_{ratio_test}_th_{ransac_th}_mnn_scale_{scaling_factor} {max_kpts}"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    logger.info(f"\n\n>>> Running parallel benchmark for {key}...<<<\n")

    # create if not exists
    results_path = Path(f"benchmarks_2D/{benchmark_name}/results")
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(results_path / "results.json"):
        with open(results_path / "results.json", "w") as f:
            json.dump({}, f)

    with open(results_path / "results.json", "r") as f:
        data = json.load(f)

    if key in data:
        results = data[key]
        import warnings

        warnings.warn("A similar run already exists.", UserWarning)

    # Define the benchmark
    benchmark = Benchmark(
        benchmark_name=benchmark_name,
        dataset_path=data_path,
        ransac_th=ransac_th,
        min_score=min_score,
        ratio_test=ratio_test,
        max_kpts=max_kpts,
        njobs=njobs,
        seed=seed,
        oom_safe=oom_safe,
        scaling_factor=scaling_factor,
        ghr_partial=ghr_partial,
        feature_path=feature_path,
        reuse_kpts_path=args.reuse_kpts,
        save_descriptors=args.save_descriptors,
        compute_repeatability=compute_repeatability,
        device=device,
    )

    # Run the benchmark
    s = time.time()
    # Create a simpler save key for features (wrapper_name + max_kpts)

    if args.run_tag is not None:
        feature_save_key = f"{wrapper.name}_{args.run_tag}_kpts_{max_kpts}"
    else:
        feature_save_key = f"{wrapper.name}_kpts_{max_kpts}"
    results, timestamp = benchmark.benchmark(wrapper, save_key=feature_save_key)
    print_metrics(wrapper, results)
    print("-------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f)

    logger.info(
        f"Results computed in {time.time() - s:.1f}s and saved to {results_path / 'results.json'}\n\n"
    )
