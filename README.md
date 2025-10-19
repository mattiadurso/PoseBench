# PoseBench

PoseBench is a lightweight suite of local wrappers for feature detection and description. It downloads third-party implementations and implements a unified interface, allowing users to test and compare methods quickly. Exact results might slightly change according to different library versions, hardware, different num
ber of jobs, or other unknown factors.

Currently it's possible to run a model/wrapper on **8** different benchmarks.

SANDesc is supported but not released yet; thus, those parts might be commented or never used.

⚠️⚠️⚠️ Repo still under development ⚠️⚠️⚠️
## Quick Start

### 1) Create the Environment

Set up the environment as follows. Other library versions might work as well, I tested the code with these.
```bash
# Create conda environment
conda create -n keypoint_factory python=3.10.16
conda activate keypoint_factory                  # Or . ./activate_env.sh if you are lazy

# Install PyTorch with CUDA support (tested with CUDA 12.4)
pip install \
  torch==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

pip install \
  h5py==3.13.0 \
  joblib==1.4.2 \
  numpy==1.26.4 \
  opencv-python==4.11.0.86 \
  pandas==2.2.3 
  

## Suggested but optional. 
# kornia: need for "disk-kornia" method. If not installed you need to provide/download at least one method in methods/ to run any benchmark
# matplotlib: used for demo and plotting validation results in read_results.ipynb
# nvidia-ml-py: to measure VRAM usage in speed and memory benchmark
# PIL: used in some visualizations, but not strictly needed for benchmarking
# pydegensac: enables better geometric estimation in our imc and hpatches implmentations. Might lead to higher performance
# tqdm: used to nicely display loops bars progression
# xformers: to increase speed when using transformer-based models (e.g., DeDoDe, RDD)
pip install \
  kornia==0.8.0 \
  matplotlib==3.10.1 \
  nvidia-ml-py==13.580.82 \
  Pillow==11.1.0 \
  pydegensac \
  tqdm==4.67.1 \
  xformers==0.0.29.post2 

# To run IMC, these are also needed
pip install \
  jsmin \
  matplotlib \
  pydegensac \
  schema \
  scipy \
  shortuuid \
  tqdm
  
```
Other dependencies might be related to third party specific methods. 

### 2) Download the Wrappers and Benchmarks

Edit `download_wrappers.py` to choose which feature extractor to download. An empty list means all methods listed in `download_wrappers.yaml`. Then, to download __all__ benchmark data and/or code, run the following:

```bash
python download_wrappers.py && \
bash bash/download_all.sh
```
To download only one benchmark, use the corresponding bash file in ```bash/```. The deafult wrapper is `disk-kornia`, which is already available when installing Kornia, and can be used to test if everything works.

### 3) Test in the Notebook
In `demo.ipynb`, it is possible to test the wrappers on images from the Graz High-Resolution Benchmark, visualizing keypoints/matches and sanity-checking that everything works.

## Feature Extraction Methods

Currently, the following methods are supported with a wrapper:

#### **SIFT**
- **[Paper](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)**: David Lowe — *Distinctive Image Features from Scale-Invariant Keypoints*
- **[Implementation](https://github.com/colmap/pycolmap)**: PyCOLMAP (provides bindings for extracting/matching SIFT features via Python; supports CPU by default (quite slow), optional CUDA).

#### **SuperPoint**
- **[Paper](https://arxiv.org/abs/1712.07629)**: Daniel DeTone, Tomasz Malisiewicz & Andrew Rabinovich — *SuperPoint: Self-Supervised Interest Point Detection and Description* (CVPR 2018 workshop; arXiv 2017)
- **[Implementation](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/)**: From the SuperGlue GitHub repository.

#### **DISK**
- **[Paper](https://arxiv.org/abs/2006.13566)**: Michał J. Tyszkiewicz, Pascal Fua & Eduard Trulls — *DISK: Learning Local Features with Policy Gradient* (NeurIPS 2020)
- **[Implementation](https://github.com/cvlab-epfl/disk)**: Official EPFL CVLAB GitHub repository containing training and inference code.

#### **RIPE**
- **[Paper](https://arxiv.org/abs/2507.04839)**: Fraunhofer HHI team — *RIPE: Reinforcement Learning on Unlabeled Image Pairs* (ICCV 2025)
- **[Implementation](https://github.com/fraunhoferhhi/RIPE)**: Fraunhofer HHI GitHub repository.

#### **DeDoDe**
- **[Paper](https://arxiv.org/abs/2308.08479)**: Johan Edstedt, Georg Bökman, Mårten Wadenbäck & Michael Felsberg — *DeDoDe: Detect, Don’t Describe — Describe, Don’t Detect for Local Feature Matching* (arXiv 2023)
- **[Implementation](https://github.com/Parskatt/DeDoDe)**: Parskatt’s GitHub repository with code, training scripts, and pretrained weights.
- **Note:** Both -B and -G descriptor models proposed in the paper are available. Repeatability results might slightly change since -G expects images to have edges multiple of 14.

#### **ALIKED**
- **[Paper](https://arxiv.org/abs/2304.03608)**: Xiaoming Zhao et al. — *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation* (2023)
- **[Implementation](https://github.com/Shiaoming/ALIKED)**: Shiaoming’s GitHub repo for the Python version.

## Supported Benchmarks

After downloading the target methods and verifying that the wrapper exists and runs correctly, the following benchmarks are supported:

### Benchmarks_2D: 

- Graz High-Resolution Benchmark
- MegaDepth-1500
- MegaDepth-View 
- Megadepth Air-to-Ground
- ScanNet-1500
- HPatches
- Image Matching Challenge (Phototourism)
- Speed and Memory

Go to [benchmarks_2D/README.md](benchmarks_2D/README.md) for more details on each of them.

### Benchmarks_3D:

- Scene Poses Estimation 

Go to [benchmarks_3D/README.md](benchmarks_3D/README.md) for more details on each of them.

### Why This Repo?

I couldn’t find a single, unified, and reproducible way to **benchmark feature extractors and poses** quickly. Setting up fair benchmarks shouldn’t steal time from research—so this repo aims to make it fast and consistent.

### Core Idea

Wrap each method with a **thin adapter** that standardizes I/O between the model and the benchmark:

* The **wrapper** handles everything the model needs for input:
  normalization, resizing/cropping/padding, color space conversion, etc.
* It produces a **standard output format**, so benchmarks can treat all methods uniformly.
* This makes swapping methods trivial—change a single argument instead of rewriting code.

#### How to Add Your Method

1. **Place your implementation** in `methods/`
2. **Write its wrapper** in `wrappers/`

   * Do all preprocessing here
   * Return outputs in the repo’s standard format (see other wrappers)
3. **Register it** in `wrappers_manager.py`

That’s it, you’re ready to benchmark.

### Benefits

* **Reproducible**: consistent I/O and evaluation across methods
* **Simple to use**: swap methods via a flag
* **Extensible**: add new models with small, focused wrappers


## TODO 

#### MISC
* [ ] Reduce dependencies
    - __pandas__ is used only in Hpatches and read_results, can be eventually put as optional and handled as list of dicts all with same keys

#### Benchmarks
* [ ] IMC
    - Add multiview support, now only stereo
* [ ] Add support for matchers (LoFTR, RoMA, etc) by changing feature extraction method and separating kpts/depth dicts. No need to repead depth extraction.
* [ ] Enable partial pose estimation with images in input scene (due to subsample).



## License and Attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.

Part of the repo is based on [Emanuele Santellani](https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en)'s work.
