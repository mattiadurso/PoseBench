## 📊 Supported Benchmarks

Once the target methods are downloaded and their wrappers are verified, the following benchmarks are available for evaluation. 

*Note: You may need to execute `gen_views_from_pycolmap.ipynb` prior to running certain benchmarks to generate the necessary view graphs.*

---

### Graz4K
**Graz4K (High-Resolution Benchmark)** is a dataset designed to evaluate feature extractors and reconstruction models under high-resolution conditions, deliberately stressing computational and memory limits. It contains six urban scenes recorded in 4K at 30 fps (sampled at 1 fps) using pre-calibrated cameras. Sparse reconstructions built with COLMAP achieved a mean reprojection error of ~0.97 px across 1.3M 3D points. After pruning view graphs and filtering pairs, the final benchmark includes 1,866 images and 4,413 image pairs. The benchmark can be run at diferent resolutions, such as 4K (3840×2160), QHD (2560×1440), or FHD (1920×1080). Results are computed following the MegaDepth-1500 protocol.

**Metrics Computed:**
* **Number of Inliers:** The raw count of inlier correspondences found during pose estimation.
* **Relative Pose Estimation AUC:** The Area Under the Curve (AUC) of the percentage of correctly estimated camera poses across angular error thresholds (5°, 10°, 20°). A pose is considered correct if both translation and rotation errors fall below the threshold.

**Run Command:**
```bash
python benchmarks_2D/benchmark_parallel.py --benchmark-name ghr 
```

---

### MegaDepth-1500
[MegaDepth-1500](https://arxiv.org/abs/2104.00680) is a curated subset of the MegaDepth dataset designed to maintain a uniform covisibility ratio across image pairs (in contrast to the Gaussian distribution seen in IMC). To penalize failures, a worst-case score of 180° is assigned when essential matrix recovery fails or the angular error exceeds 10°. For fair comparison, we recommend evaluating methods using keypoint budgets of 2K and 30K. *(Hardware note: On an RTX 4090 using 16 cores, SuperPoint completes this benchmark in under one minute).*

**Metrics Computed:** Number of Inliers and Relative Pose Estimation AUC.

**Run Command:**
```bash
python benchmarks_2D/benchmark_parallel.py --benchmark-name md  
```

---

### MegaDepth-View 
The [MegaDepth-View](https://arxiv.org/abs/2505.08013) test set is derived from MegaDepth test scenes (Internet photos with COLMAP poses and MVS depths). Image pairs are mined from existing MegaDepth reconstructions via bi-directional warping using known poses and depths. Pairs retaining 2K–20K matching pixels are kept, yielding 1,487 challenging pairs that emphasize large viewpoint and scale changes.

**Metrics Computed:** Number of Inliers and Relative Pose Estimation AUC.

**Run Command:**
```bash
python benchmarks_2D/benchmark_parallel.py --benchmark-name mdv  
```

---

### MegaDepth Air-to-Ground
The [MegaDepth Air-to-Ground](https://arxiv.org/abs/2505.08013) test set consists of images collected from Internet drone videos and ground photos across 41 landmarks. Frames extracted from the drone videos are jointly reconstructed with ground images via COLMAP to obtain camera poses and depths (~27k images, >600k candidate pairs). Depth maps are post-processed (masking sky, vehicles, and people via ADE20K segmentation, and removing small/isolated regions) to improve warping quality. The final set comprises 1,500 randomly selected pairs, validated via the same warping-based overlap test used in MegaDepth-View.

**Metrics Computed:** Number of Inliers and Relative Pose Estimation AUC.

**Run Command:**
```bash
python benchmarks_2D/benchmark_parallel.py --benchmark-name mda  
```

---

### ScanNet-1500
[ScanNet-1500](https://arxiv.org/abs/1911.11763) (SC1500) is a curated benchmark derived from the ScanNet dataset, built specifically to evaluate wide-baseline indoor image matching. Unlike earlier works that select pairs based on temporal proximity or SfM covisibility, SC1500 uses an overlap score computed directly from ground-truth poses and depth. This produces significantly more diverse and challenging image pairs. The benchmark consists of 1,500 test pairs spanning a variety of scene geometries and viewpoints.

**Metrics Computed:** Number of Inliers and Relative Pose Estimation AUC. *(Repeatability is excluded as absolute poses are unavailable).*

**Run Command:**
```bash
python benchmarks_2D/benchmark_parallel.py --benchmark-name sc  
```

---

### HPatches
[HPatches](https://arxiv.org/abs/1704.05939) is a standard benchmark containing image sequences with notable viewpoint or illumination changes. Evaluation spans 108 scenes, each featuring one reference and five target images paired by ground-truth homographies. Testing utilizes a fixed budget of 2048 keypoints and MNN for matching, strictly following the [S-TREK](https://arxiv.org/abs/2308.14598) and [D2-Net](https://arxiv.org/abs/1905.03561) protocols.

**Metrics Computed:**
* **Repeatability:** Ratio of repeated keypoints between an image pair (after applying the known homography) relative to the smaller number of keypoints detected in the two images.
* **Mean Matching Accuracy (MMA):** Percentage of matches whose reprojection error falls within predefined pixel thresholds (1, 2, and 3 pixels).
* **Matching Score:** Ratio of correct matches (within a pixel threshold) to the average number of keypoints in the overlapping image area.
* **Homography Accuracy (AUC):** Area under the curve of the percentage of estimated homographies where the corner error is below ε. Corner error is the average distance between the four reference corners and the warped target corners. The best score across multiple RANSAC thresholds is reported.

**Run Command:**
```bash
python benchmarks_2D/hpatches/hpatches_benchmark.py 
```

---

### Image Matching Challenge (Phototourism)
The [Image Matching Challenge 2021](https://github.com/ubc-vision/image-matching-benchmark) (IMC) evaluates local feature matching in complex, real-world environments. PoseBench supports the Phototourism test set, encompassing nine scenes of 100 tourist photos each, captured under diverse cameras, viewpoints, and lighting conditions. Images within a scene are exhaustively compared following the official protocol. *(Note: Currently, only stereo matching is supported, evaluated at budgets of 2,048 and 8,000 keypoints).*

*Performance Note: Although this benchmark is heavily parallelized, evaluation takes approximately 1 hour per method, reflecting its comprehensive and exhaustive nature.*

**Metrics Computed:**
* **Repeatability:** Reported at a 3-pixel threshold.
* **Number of Inliers:** Raw count of inlier correspondences.
* **Relative Pose Estimation AUC:** Computed at angular error thresholds of 5° and 10°. A failure is assigned when the error exceeds 10°.

**Run Command:**
```bash
python benchmarks_2D/imc/imc_benchmark.py 
```

---

### Speed and Memory
The **Speed and Memory** benchmark assesses the computational efficiency of feature extraction methods. By measuring practical resource consumption, this benchmark determines the feasibility of deploying different methods in real-world applications where strict processing and memory constraints exist.

The benchmark runs each method on a predefined set of images using a highly consistent environment to guarantee fair comparisons.

**Metrics Computed:**
* **Runtime:** The average time taken to process a single image or image pair, measured in milliseconds (ms).
* **Memory Usage:** The peak VRAM/RAM consumption during feature extraction, measured in megabytes (MB).

**Run Command:**
```bash
python benchmarks_2D/speed_and_memory/benchmark_speed.py
```