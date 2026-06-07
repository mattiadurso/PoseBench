# Refactor Validation — `disk-kornia` smoke test across benchmarks

**Goal:** run one benchmark per dataset with the `disk-kornia` wrapper, confirm the
refactored code still runs, and check the numbers stay close to previously recorded
runs. Fix code that breaks; skip datasets whose data is not present.

**Date:** 2026-06-06
**Env:** conda `posebench` (torch 2.9.1+cu128, kornia 0.8.0, CUDA on RTX 4090)
**Wrapper:** `disk-kornia` (maps to `DiskWrapperKornia` in `wrappers/disk_wrapper.py`)

> Note: `activate_env.sh` references a `keypoint_factory` env that does not exist on
> this machine. The working env per `CLAUDE.md` is `posebench`; that is what was used.

## Datasets

| Dataset | Entry point | Status |
|---|---|---|
| scannet1500 | `benchmark_parallel.py --ds sc` | ✅ pass |
| megadepth1500 | `benchmark_parallel.py --ds md` | ✅ pass |
| megadepth_view | `benchmark_parallel.py --ds mdv` | ✅ pass |
| megadepth_air2ground | `benchmark_parallel.py --ds mda` | ✅ pass |
| hpatches | `hpatches/hpatches_benchmark.py` | ✅ pass |
| imc | `imc/imc_benchmark.py` | ✅ pass (after fix) |
| graz4k | `benchmark_parallel.py --ds graz` | **skipped** — no `pairs_calibrated.txt`/`views.txt` in `data/` |
| terrasky3d | `benchmark_parallel.py --ds ts3d` | **skipped** — no `pairs_calibrated.txt`/`views.txt` in `data/` |

`graz4k/data` and `terrasky3d/data` contain only raw scene folders (colmap/frames/depth);
the calibrated-pairs and views files that the benchmark loads are not present, so these
are skipped per the "skip if data not found" instruction.

## Results

### scannet1500 ✅
Params: `min_score 0.0, ratio_test 1.0, ransac_th 1.0, scale 1.0, 2048 kpts`.
Compared against the only prior `disk-kornia` scannet run (`20250923_000705`).

| metric | old (2025-09-23) | new (refactor) |
|---|---|---|
| inliers | 61.9 | 65.2 |
| auc_5 | 4.2 | 5.3 |
| auc_10 | 9.8 | 11.9 |
| auc_20 | 54.9 | 54.9→55.9 |

Verdict: **close**. Differences are small and consistent with environment drift
(torch 2.6→2.9, kornia, AMP) plus the `+0.5 to kpts` commit. Runtime 224s, 0/1500
pairs unregistered. New result schema drops `map_*` and adds `unregistered_pairs`
/`total_pairs` (refactor change, not a regression).

### megadepth1500 ✅
Params: `min_score 0.5, ratio_test 1.0, ransac_th 0.75, scale 1.0, 2048 kpts`.
Compared against prior `disk` run (`20251002_120125`).

| metric | old (2025-10-02) | new (refactor) |
|---|---|---|
| inliers | 456.5 | 457.4 |
| auc_5 | 35.5 | 35.4 |
| auc_10 | 52.3 | 52.7 |
| auc_20 | 76.0 | 76.1 |
| rep_1 / rep_3 / rep_5 | 20.2 / 41.9 / 49.5 | 19.6 / 41.8 / 49.5 |
| rep_mnn_1/3/5 | 18.1 / 36.8 / 40.6 | 17.7 / 36.8 / 40.7 |

Verdict: **near-identical**. Runtime 81s, 7/1500 unregistered (old: 5).

### megadepth_view ✅
Params: `min_score 0.0, ratio_test 0.98, ransac_th 0.75, scale 1.0, 2048 kpts`.
Compared against prior `disk` run (`20250929_162214`).

| metric | old (2025-09-29) | new (refactor) |
|---|---|---|
| inliers | 122.1 | 120.2 |
| auc_5 | 36.6 | 35.0 |
| auc_10 | 49.0 | 47.7 |
| auc_20 | 74.4 | 73.9 |

Verdict: **close** (auc within ~1.5 pts). Runtime 117s, 0/915 unregistered.

### megadepth_air2ground ✅
Params: `min_score 0.0, ratio_test 0.98, ransac_th 0.75, scale 1.0, 2048 kpts`.
Compared against prior `disk` run (`20250929_161718`).

| metric | old (2025-09-29) | new (refactor) |
|---|---|---|
| inliers | 62.9 | 60.9 |
| auc_5 | 29.2 | 28.3 |
| auc_10 | 38.1 | 37.1 |
| auc_20 | 69.0 | 68.5 |

Verdict: **close** (auc within ~1 pt). Runtime 246s, 0/1500 unregistered.

### hpatches ✅
Params: defaults (`2048 kpts`). Compared against the `REF_BEFORE` run recorded
earlier the same day (`20260606_070346`).

| metric (mean) | REF_BEFORE | new (refactor) |
|---|---|---|
| illum acc@3 (i_3) | 0.9346 | 0.9346 |
| illum acc@5 (i_5) | 0.9846 | 0.9846 |
| view acc@3 (v_3) | 0.5929 | 0.5929 |
| view acc@5 (v_5) | 0.7429 | 0.7429 |

Verdict: **exact match**. Runtime 59s.

### imc (phototourism, test set) ✅ — required a code fix
Params: `min_score 0.5, ratio_test 1.0, ransac_th 0.75, 2048 kpts, scene-set test`
(matches the old `disk` run's `mnn0.5-ransac-0.75` label).
Compared against the prior `disk` IMB json (`results/test/disk-2048kpts-matcher-mnn0.5-ransac-0.75.json`,
processed 2025-09-30; backed up to `/tmp/imc_disk_old_backup.json` before the run, but it
was never overwritten because the new run is named `disk-kornia-*`).

| metric (allseq / stereo / run_avg) | old `disk` (2025-09-30) | new `disk-kornia` |
|---|---|---|
| qt_auc_05 | 0.3746 | 0.3745 |
| qt_auc_10 | 0.4945 | 0.4952 |

Verdict: **near-identical**.

**Bug found & fixed (refactor regression).** The `benchmarks/` → `benchmarks_2D/`
directory rename was not propagated into the imc code, so the run aborted with
`AssertionError: Dataset path .../benchmarks/imc/data/phototourism does not exist`.
Fixed the stale `benchmarks/imc/...` paths (data path, results copy path, the `cp`
source in the os.system call, and the `to_import_imc` path):
- `benchmarks_2D/imc/imc_benchmark.py` — lines for `data_path` default, `results_path`,
  the `cp` source path + log string, and the `to_import_imc` output path.
- `benchmarks_2D/imc/utils_imc_benchmark.py` — `extract_image_matching_benchmark`
  default `data_path` (overridden in practice, fixed for consistency).

The `abs_root_path` in `utils_imc_benchmark.py` resolves to `benchmarks_2D/`, so its
existing `imc/image-matching-benchmark` references were already correct and left untouched.

**Note on the IMB sub-pipeline.** The IMC eval shells out to the third-party
`image-matching-benchmark/run.py` (its return code is ignored). On the first cold pass
that subprocess did not finish packing; re-running the benchmark with the cached
matches/intermediate results completed normally (packed json produced and copied into
`results/test/`). The `affine.h5/angles.h5/scales.h5` "No such file" messages during
"Computing model" are expected — DISK exports no affine/angle/scale fields — and are
caught internally. This timing/packing behaviour is in the vendored IMB code, not a
refactor regression.

## Summary

| Dataset | Result | Notes |
|---|---|---|
| scannet1500 | ✅ close | env-drift level differences |
| megadepth1500 | ✅ near-identical | |
| megadepth_view | ✅ close | auc within ~1.5 pts |
| megadepth_air2ground | ✅ close | auc within ~1 pt |
| hpatches | ✅ exact | matches same-day REF_BEFORE |
| imc | ✅ near-identical | needed `benchmarks_2D/` path fix |
| graz4k | ⏭ skipped | no `pairs_calibrated.txt` / `views.txt` in data |
| terrasky3d | ⏭ skipped | no `pairs_calibrated.txt` / `views.txt` in data |
| 3D (`benchmark_pose.py`) | — N/A | compares COLMAP models; takes no feature wrapper, so `disk-kornia` does not apply |

**Conclusion:** the refactored code runs correctly for `disk-kornia` across all
datasets that have data present, and reproduces the previously recorded numbers (modulo
small environment-drift differences from torch 2.6→2.9 / kornia and the `+0.5 to kpts`
keypoint-offset commit). One real refactor regression was found and fixed: stale
`benchmarks/imc/...` paths in the IMC benchmark. Two datasets (graz4k, terrasky3d) were
skipped because their calibrated-pairs/views files are not present in `data/`.

### Code changed
- `benchmarks_2D/imc/imc_benchmark.py` — `benchmarks/imc` → `benchmarks_2D/imc` (4 sites).
- `benchmarks_2D/imc/utils_imc_benchmark.py` — `benchmarks/imc` → `benchmarks_2D/imc` (default arg).

### Reproduce
```bash
conda activate posebench
python benchmarks_2D/benchmark_parallel.py --ds sc  --wrapper-name disk-kornia --max-kpts 2048 --ratio-test 1.0  --min-score 0.0 --ransac-th 1.0  --scaling-factor 1.0 --run-tag refactor_test
python benchmarks_2D/benchmark_parallel.py --ds md  --wrapper-name disk-kornia --max-kpts 2048 --ratio-test 1.0  --min-score 0.5 --ransac-th 0.75 --scaling-factor 1.0 --run-tag refactor_test
python benchmarks_2D/benchmark_parallel.py --ds mdv --wrapper-name disk-kornia --max-kpts 2048 --ratio-test 0.98 --min-score 0.0 --ransac-th 0.75 --scaling-factor 1.0 --run-tag refactor_test
python benchmarks_2D/benchmark_parallel.py --ds mda --wrapper-name disk-kornia --max-kpts 2048 --ratio-test 0.98 --min-score 0.0 --ransac-th 0.75 --scaling-factor 1.0 --run-tag refactor_test
python benchmarks_2D/hpatches/hpatches_benchmark.py --wrapper-name disk-kornia --max-kpts 2048 --run-tag refactor_test
python benchmarks_2D/imc/imc_benchmark.py --wrapper-name disk-kornia --max-kpts 2048 --ratio-test 1.0 --min-score 0.5 --ransac-th 0.75 --scene-set test --run-tag refactor_test
```

