## Supported Benchmarks

After downloading the target methods and verifying that the wrapper exists and runs correctly, the following benchmarks are supported.

### Scene Pose Estimation

Given one or more reconstructed scenes and their corresponding ground truth (GT), the benchmarks compute relative poses exhaustively and compare them against the GT.

When evaluating a single scene, provide the paths to the reconstruction and ground-truth folders:

```bash
python benchmarks_3D/benchmark_pose.py \
    --input-model /results/eth3d/botanical_garden/sparse_vggt \
    --target-model /dataset/eth3d/botanical_garden/sparse_gt
```

When evaluating multiple scenes, point to the parent directories and specify the model subfolders:
```bash
python benchmarks_3D/benchmark_pose.py \
    --input-model /results/eth3d  --input-folder sparse_vggt \
    --target-model /dataset/eth3d --target-folder sparse_gt
```

If the sets of scenes are not identical, only the common scenes are tested.

NOTICE: 
- Models need to be in COLMAP format (cameras, images, points3D)
- Images names need to correspond, e.g. both include folder or not.
