### Scene Pose Estimation (3D Benchmark)

This benchmark evaluates 3D scene reconstruction by exhaustively computing relative poses from one or more reconstructed scenes and comparing them against their corresponding Ground Truth (GT).

**Evaluating a Single Scene:**
When evaluating an individual scene, provide the direct paths to the reconstruction and ground-truth directories:
```bash
python benchmarks_3D/benchmark_pose.py \
    --input-model /results/eth3d/botanical_garden/sparse_vggt \
    --target-model /dataset/eth3d/botanical_garden/sparse_gt
```

**Evaluating Multiple Scenes (Batch Processing):**
When evaluating multiple scenes simultaneously, point to the parent directories and specify the respective model subfolders. *(Note: If the input and target directories contain mismatched sets of scenes, the benchmark will automatically filter and evaluate only the overlapping/common scenes).*
```bash
python benchmarks_3D/benchmark_pose.py \
    --input-model /results/eth3d  --input-folder sparse_vggt \
    --target-model /dataset/eth3d --target-folder sparse_gt
```

**⚠️ Important Requirements:**
* **COLMAP Format:** All input and target models must be in the standard COLMAP format (containing `cameras`, `images`, and `points3D` files).
* **Filename Consistency:** Image names must match exactly between the reconstruction and the ground truth. Specifically, path formats must be identical (e.g., both must either include or omit directory prefixes).