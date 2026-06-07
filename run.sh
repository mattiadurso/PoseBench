#!/bin/bash


# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name aliked --scaling-factor 2 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75 \
#     --custom-desc /home/mattia/Desktop/Repos/posebench/sandesc_models/aliked/final.pth --sandesc-paper 


python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/HDD_Slow/TerraSky3D/data_test --wrapper-name superpoint --scaling-factor 1 --max-kpts 2048 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75

# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name aliked --scaling-factor 2 --max-kpts 2048 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75 \
#     --custom-desc /home/mattia/Desktop/Repos/sandesc/wandb/run-20260221_182957-4dc70px2/files/saved_model/checkpoint-iteration-20480.pth  --sandesc-paper 



# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name superpoint --scaling-factor 2 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name disk-kornia --scaling-factor 2 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name dedode-G --scaling-factor 2 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name rdd --scaling-factor 2.1 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name aliked --scaling-factor 2 --max-kpts 4096 --ratio-test 0.98 --min-score 0.5 --ransac-th 0.75


# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name loftr --ransac-th 0.75 --max-kpts 4096
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name disk+lightglue --ransac-th 0.75 --max-kpts 4096  
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name aliked+lightglue --ransac-th 0.75 --max-kpts 4096  
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name mast3r --ransac-th 0.75 --max-kpts 4096
# python benchmarks_2D/benchmark_parallel.py --ds ts3d --data-path /home/mattia/Desktop/datasets/mydataset/data_test --wrapper-name roma --ransac-th 0.75 --max-kpts 4096  

# python benchmarks_2D/benchmark_parallel.py --ds md --wrapper-name aliked --ransac-th 0.75 --ratio-test 0.98 --min-score 0.5 --max-kpts 2048 --custom-desc /home/mattia/Desktop/Repos/sandesc/wandb/run-20260606_074520-y2tpn0dc/files/saved_model/checkpoint-iteration-17408.pth --run-tag refactor
