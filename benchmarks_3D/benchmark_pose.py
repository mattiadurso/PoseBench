"""3D scene pose-estimation benchmark comparing COLMAP reconstructions."""

from __future__ import annotations

import os
import time
import logging
import pycolmap
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathlib import Path
from tqdm.auto import tqdm
from itertools import combinations

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils_benchmark_pose import compute_AUC, evaluate_R_t

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _scene_image_names(rec):
    """Return the sorted array of image names in a reconstruction."""
    return np.array(sorted([img.name for img in rec.images.values()]))


def _pose_from_rec(rec, image_path):
    """Return the (R, t) cam_from_world pose of an image by name."""
    img = rec.find_image_with_name(image_path)
    return img.cam_from_world.rotation.matrix(), img.cam_from_world.translation


def _relative_pose_error(target_rec, input_rec, image_1_path, image_2_path, deg):
    """Relative-pose error between two images, in target vs input models.

    Returns:
        (q_error, t_error, max_error); max_error is capped to inf above 10.
    """
    R1_target, t1_target = _pose_from_rec(target_rec, image_1_path)
    R2_target, t2_target = _pose_from_rec(target_rec, image_2_path)
    R1_input, t1_input = _pose_from_rec(input_rec, image_1_path)
    R2_input, t2_input = _pose_from_rec(input_rec, image_2_path)

    # relative pose target / input
    R_target = R2_target @ R1_target.T
    t_target = t2_target - R_target @ t1_target
    R_pred = R2_input @ R1_input.T
    t_pred = t2_input - R_pred @ t1_input

    q_err, t_err = evaluate_R_t(R_pred, t_pred, R_target, t_target, deg=deg)
    max_error = max(q_err, t_err)
    max_error = max_error if max_error < 10 else np.inf
    return q_err, t_err, max_error


def evaluate_scene(target_rec, input_rec, deg=True, verbose=False):
    """
    Given two dictionaries {"image_idx":{qvec, tvec}}, evaluate the relative pose between all the possible pairs of images.
    Args:
        images_target_dict:   dictionary with the ground truth poses.
        images_pred_dict: dictionary with the predicted poses.
        deg: if True, the errors are returned in degrees else in radians. Default is True.
    Returns:
        df: dataframe with the relative pose errors with keys {image1, image2, q_error, t_error}.
        num_images: number of images in the target set.
    """

    df = {
        "image1": [],
        "image2": [],
        "q_error": [],
        "t_error": [],
        "max_error": [],
    }
    target_images = _scene_image_names(target_rec)
    input_images = _scene_image_names(input_rec)

    # for each pair of images in the ground truth
    for image_1_path, image_2_path in combinations(target_images, 2):
        if (image_1_path in input_images) and (image_2_path in input_images):
            q_err, t_err, max_error = _relative_pose_error(
                target_rec, input_rec, image_1_path, image_2_path, deg
            )
        else:
            q_err, t_err, max_error = np.inf, np.inf, np.inf
            if verbose:
                logger.info(
                    f"Image {image_1_path} or {image_2_path} not in input model."
                )

        # append to the dataframe
        df["image1"].append(image_1_path)
        df["image2"].append(image_2_path)
        df["q_error"].append(q_err)
        df["t_error"].append(t_err)
        df["max_error"].append(max_error)

    return pd.DataFrame(df), (len(input_images), len(target_images))


def eval_colmap_model(
    model_path,
    target_path,
    thrs=[1, 3, 5],
    return_df=False,
    AUC_col="max_error",
    verbose=False,
):
    """
    Given a scene path, evaluate the model in the given folder versus the ground truth in the target_folder.
    Args:
        model_path: path to the model to evaluate.
        target_path:    path to the ground truth model.
        thrs:       list of thresholds for the AUC computation.
        return_df:  if True, return the dataframe with the errors.
        AUC_col:    column to compute the AUC. Default is "max_error". Other options are "q_error" and "t_error".
    Returns:
        df_AUC: dataframe with the AUC values for the model.

    """
    # read model
    try:
        rec_input = pycolmap.Reconstruction(model_path)
    except Exception as e:
        print(f"Failed to read input model from {model_path}: {e}")
        return np.array([np.nan] * len(thrs)), (np.nan, np.nan), None

    try:
        rec_target = pycolmap.Reconstruction(target_path)
    except Exception as e:
        print(f"Failed to read target model from {target_path}: {e}")
        return np.array([np.nan] * len(thrs)), (np.nan, np.nan), None

    # evaluate scene (each pair of images) and compute the AUC
    df, num_images = evaluate_scene(rec_target, rec_input, verbose=verbose)
    AUC_score_max = np.array(compute_AUC(df[AUC_col], thrs))

    if return_df:
        return AUC_score_max, num_images, df

    return AUC_score_max, num_images, None


def _valid_scene_pairs(common_scenes, input_path, target_path, input_folder, target_folder):
    """Return (valid_pairs, valid_scenes) for scenes whose model paths exist."""
    valid_pairs = []
    valid_scenes = []
    for scene in common_scenes:
        inp = os.path.join(input_path, scene, input_folder)
        tgt = os.path.join(target_path, scene, target_folder)
        if os.path.exists(inp) and os.path.exists(tgt):
            valid_pairs.append((inp, tgt))
            valid_scenes.append(scene)
        else:
            logger.warning(f"Skipping {scene}: paths don't exist at {inp} and {tgt}")
    return valid_pairs, valid_scenes


def _save_scene_dfs(input_path, valid_scenes, dfs):
    """Write each scene's per-pair error dataframe to a CSV directory."""
    dfs_path = Path(input_path + "_results_dfs")
    dfs_path.mkdir(parents=True, exist_ok=True)
    for scene_name, df in zip(valid_scenes, dfs):
        if df is not None:
            df.to_csv(dfs_path / f"results_{scene_name}.csv", index=False)
    print(f"Saved individual result dataframes to {dfs_path}")


def _assemble_scene_auc_df(results, valid_scenes, reg_images, tot_images, thrs, round_to):
    """Build the per-scene AUC table with image counts and a trailing mean row."""
    res = {}
    for auc_scores, scene_name in zip(results, valid_scenes):
        if auc_scores is not None:
            res[scene_name] = auc_scores

    # transpose to have the scenes as rows
    df_res_colmap = pd.DataFrame(res, index=thrs).transpose()
    df_res_colmap["reg_images"] = reg_images
    df_res_colmap["tot_images"] = tot_images
    df_res_colmap = df_res_colmap.sort_index()

    df_res_colmap.columns = [f"auc@{thr}" for thr in thrs] + [
        "reg_images",
        "tot_images",
    ]
    df_res_colmap = df_res_colmap[
        ["reg_images", "tot_images"] + [f"auc@{thr}" for thr in thrs]
    ]
    df_res_colmap.loc["mean"] = df_res_colmap.mean(numeric_only=True)
    return df_res_colmap.round(round_to)


def eval_colmap_model_all_scenes(
    input_path,
    target_path,
    input_folder="colmap/sparse/0",
    target_folder="sparse",
    thrs=[0.5, 1, 3, 5, 10],
    AUC_col="max_error",
    return_df=False,
    n_jobs=-1,
    round_to=2,
    verbose=True,
) -> pd.DataFrame:
    """
    Evaluate the model on all the scenes in the data_path using parallel processing.
    These must be in COLMAP format. The model is evaluated at the specified thresholds.
    """

    # Keep only scenes present in both directories
    common_scenes = sorted(set(os.listdir(input_path)) & set(os.listdir(target_path)))
    print(f"Found {len(common_scenes)} common scenes.")

    if len(common_scenes) == 0 and verbose:
        logger.warning("No common scenes found!")
        return pd.DataFrame()

    valid_pairs, valid_scenes = _valid_scene_pairs(
        common_scenes, input_path, target_path, input_folder, target_folder
    )
    print(f"Evaluating {len(valid_pairs)} valid scenes.")

    # Use joblib to parallelize the evaluation of each scene
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(eval_colmap_model)(
            input,
            target,
            thrs=thrs,
            return_df=return_df,
            AUC_col=AUC_col,
            verbose=verbose,
        )
        for input, target in tqdm(
            valid_pairs,
            desc="Evaluating scenes",
            total=len(valid_pairs),
        )
    )
    # Unpack the results
    results = [r[0] for r in parallel_results]
    num_images = [r[1] for r in parallel_results]
    reg_images = [r[0] for r in num_images]  # registered images
    tot_images = [r[1] for r in num_images]  # total images in target
    dfs = [r[2] for r in parallel_results] if return_df else None

    if return_df:
        _save_scene_dfs(input_path, valid_scenes, dfs)

    return _assemble_scene_auc_df(
        results, valid_scenes, reg_images, tot_images, thrs, round_to
    )


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to the input 3D model file or directorys. E.g., scenes/scene_name/colmap/sparse/0 or scenes/",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Path to the target 3D model file or directorys. E.g., scenes/scene_name/colmap/sparse/0 or scenes/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks_3D/results",
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--many-scenes",
        action="store_true",
        help="If set, evaluate all scenes in the input directory.",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="colmap/sparse/0",
        help="Subfolder in each scene folder where the input model is located. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        default="sparse",
        help="Subfolder in each scene folder where the ground truth model is located. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--mapper",
        type=str,
        default="colmap",
        help="Name of the mapper used to generate the input model. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--thrs",
        type=float,
        nargs="+",
        default=[1, 3, 5, 10, 15, 30],
        help="Thresholds for AUC computation.",
    )
    args = parser.parse_args()

    s = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.many_scenes:
        df_res = eval_colmap_model_all_scenes(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            n_jobs=16,
            input_folder=args.input_folder,
            target_folder=args.target_folder,
            mapper=args.mapper,
        )
        df_res.to_csv(
            os.path.join(args.output_dir, f"results_all_scenes_{args.mapper}.csv"),
            index=True,
        )
        print(df_res)
    else:
        auc_scores, df = eval_colmap_model(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            return_df=True,
        )
        print(f"AUC scores at {args.thrs}: {auc_scores}")
        df.to_csv(
            os.path.join(args.output_dir, f"results_single_scene_{args.mapper}.csv"),
            index=False,
        )
        print(df)

    print(f"Total time: {time.time() - s:.2f} seconds.")
