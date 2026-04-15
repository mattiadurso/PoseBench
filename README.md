# PoseBench

PoseBench is a lightweight, unified evaluation suite for local feature detection, description, and matching. By providing a standardized interface for third-party implementations, PoseBench allows researchers and developers to quickly test, compare, and benchmark various methods under reproducible conditions.

*Note: Exact numerical results may slightly vary depending on library versions, hardware configurations, and parallelization overhead.*

Currently, PoseBench supports executing wrappers across **8** different benchmarks.

> **🎉 Update:** SANDesc has been released! Check it out here: [SANDesc](https://github.com/mattiadurso/SANDesc)

*🚨 Note: This repository is under development.*

---

## 🚀 Quick Start

### 1. Environment Setup

Set up the environment using Conda. The codebase has been tested with the following library versions, though other recent versions may also be compatible.

```bash
# Create and activate the conda environment
conda create -n keypoint_factory python=3.10.16
conda activate posebench  # Or use: source activate_env.sh

# Install PyTorch with CUDA support (Tested with CUDA 12.4)
pip install torch==2.6.0+cu124 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# Install core dependencies
pip install \
  h5py==3.13.0 \
  joblib==1.4.2 \
  numpy==1.26.4 \
  opencv-python==4.11.0.86 \
  pandas==2.2.3 

# Install dependencies for 3D benchmarking
pip install pycolmap==3.11
````

#### Optional (but Recommended) Dependencies

Depending on your specific use cases (e.g., running specific matchers, visualizations, or specific benchmarks), you may need the following:

```bash
# Common benchmark and visualization tools
pip install \
  kornia==0.8.0 \
  matplotlib==3.10.1 \
  nvidia-ml-py==13.580.82 \
  Pillow==11.1.0 \
  pydegensac \
  tqdm==4.67.1 \
  xformers==0.0.29.post2 

# Additional requirements for the IMC Benchmark
pip install \
  jsmin \
  schema \
  scipy \
  shortuuid
```

*Note: Certain third-party methods may require additional, specific dependencies. Refer to their respective wrappers.*

### 2\. Download Methods and Data

To specify which feature extractors to download, edit `download_wrappers.py`. Leaving the list empty will download all methods defined in `download_wrappers.yaml`.

To download **all** benchmarks, datasets, and wrappers, run:

```bash
python download_wrappers.py
bash bash/download_all.sh
```

**Dataset-Specific Downloads:**
To download only a specific benchmark dataset, execute its corresponding bash script in the `bash/` directory. For example, to download the Graz4K dataset:

```bash
bash bash/download_graz4k.sh
```

*Tip: The default wrapper is `disk-kornia` (available automatically if Kornia is installed), which is excellent for verifying that your setup is working correctly.*

### 3\. Testing via Notebook

You can use `demo.ipynb` to test the wrappers on images from the Graz4K Benchmark. This notebook provides visualizations for keypoints and matches, serving as a quick sanity check to ensure your environment and wrappers are functioning properly.

-----

## 🛠️ Supported Methods

PoseBench currently provides wrappers for the following methods.

### Sparse Extractors

  * **SIFT** ([Paper](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) | [Implementation](https://github.com/colmap/pycolmap)): Leverages PyCOLMAP bindings. CPU by default; CUDA supported optionally.
  * **SuperPoint** ([Paper](https://arxiv.org/abs/1712.07629) | [Implementation](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/)): Deep feature extractor from MagicLeap.
  * **DISK** ([Paper](https://arxiv.org/abs/2006.13566) | [Implementation](https://github.com/kornia/kornia)): Policy-gradient learned features, powered by Kornia.
  * **RIPE** ([Paper](https://arxiv.org/abs/2507.04839) | [Implementation](https://github.com/fraunhoferhhi/RIPE)): Reinforcement learning approach by Fraunhofer HHI.
  * **DeDoDe** ([Paper](https://arxiv.org/abs/2308.08479) | [Implementation](https://github.com/Parskatt/DeDoDe)): Both `-B` and `-G` descriptor models are supported. *(Note: `-G` requires image dimensions to be multiples of 14).*
  * **ALIKED** ([Paper](https://arxiv.org/abs/2304.03608) | [Implementation](https://github.com/Shiaoming/ALIKED)): Deformable transformation-based lightweight keypoints.

### Deep Matchers

*Note: Deep matchers are currently supported for Graz4K, MegaDepth-1500, MegaDepth-View, Megadepth Air-to-Ground, and ScanNet-1500 benchmarks.*

  * **LoFTR** ([Paper](https://arxiv.org/abs/2104.00680) | [Implementation](https://github.com/kornia/kornia)): Detector-free matching with Transformers.
  * **LightGlue** ([Paper](https://arxiv.org/abs/2306.13643) | [Implementation](https://github.com/kornia/kornia)): Fast, adaptive local feature matching.
  * **RoMa** ([Paper](https://arxiv.org/abs/2305.15404) | [Implementation](https://github.com/Parskatt/RoMa)): Robust dense feature matching.

-----

## 📊 Supported Benchmarks

Once your methods are downloaded and verified, PoseBench allows you to evaluate them across the following tasks:

### 2D Benchmarks

For detailed information, see [benchmarks\_2D/README.md](https://www.google.com/search?q=benchmarks_2D/README.md).

  - Graz4K
  - MegaDepth-1500
  - MegaDepth-View
  - Megadepth Air-to-Ground
  - ScanNet-1500
  - HPatches
  - Image Matching Challenge (Phototourism)
  - Speed and Memory Profiling

### 3D Benchmarks

For detailed information, see [benchmarks\_3D/README.md](https://www.google.com/search?q=benchmarks_3D/README.md).

  - Scene Pose Estimation

-----

## 💡 Architecture & Philosophy

### Why PoseBench?

Setting up fair and reproducible benchmarks for feature extraction and matching is historically time-consuming. PoseBench removes the engineering overhead from your research workflow by providing a fast, consistent, and standardized evaluation suite.

### Core Architecture

PoseBench relies on a **thin adapter pattern**. It wraps each method to standardize Input/Output operations between the underlying model and the benchmark suite:

  * **Preprocessing:** The wrapper handles normalizations, resizing, cropping, padding, and color space conversions.
  * **Standardized Output:** It returns results in a uniform format, allowing benchmarking scripts to treat all methods agnostically.
  * **Swappability:** Changing the evaluated model requires changing a single command-line argument rather than rewriting pipeline logic.

### How to Add Your Own Method

Adding a custom method to PoseBench is straightforward:

1.  **Place your implementation** inside the `methods/` directory.
2.  **Create a wrapper script** in `wrappers/`. Handle all method-specific preprocessing here and ensure the output matches the standard PoseBench format (refer to existing wrappers as templates).
3.  **Register your wrapper** inside `wrappers_manager.py`.

You are now ready to benchmark your method against the state-of-the-art.



## 📜 License and Attribution

This repository provides wrapper code to interface with third-party research code and models. **Each downloaded project remains under its original author's license.** By using this tool, you are responsible for reviewing and complying with the licenses of the respective upstream authors.

Parts of this repository's benchmarking logic are based on the work of [Emanuele Santellani](https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en).


