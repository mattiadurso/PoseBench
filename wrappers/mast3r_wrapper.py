"""Wrapper for the MASt3R dense 3D-aware matcher."""

import sys
import warnings
import torch
from PIL import Image

from pathlib import Path
from typing import Union

method_path = Path(__file__).resolve().parents[1] / "methods/mast3r"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)
from wrappers.wrapper import MethodWrapper, PairMatches

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r  # noqa: F401  (side-effect: adds dust3r to sys.path)
from dust3r.inference import inference
from dust3r.utils.image import load_images


class MAST3RWrapper(MethodWrapper):
    """Wraps the pretrained AsymmetricMASt3R model as a dense pair matcher."""

    def __init__(
        self,
        device: str = "cuda:0",
        border: int = 16,
    ) -> None:
        """Load the pretrained metric AsymmetricMASt3R model.

        Args:
            device: Torch device to load the model onto.
            border: Image border margin (pixels) inherited by MethodWrapper.
        """
        name = "mast3r"
        super().__init__(name=name, border=border, device=device)
        self.is_sparse_feature_extractor = False

        # Load weights
        self.model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(device)

    def match_pair(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        max_kpts: int = 4096,
    ) -> PairMatches:
        """Match an image pair end-to-end with MASt3R."""
        images = load_images([str(img1_path), str(img2_path)], size=512, verbose=False)

        # Get original image sizes
        img1_pil = Image.open(img1_path)
        img2_pil = Image.open(img2_path)
        W1, H1 = img1_pil.size
        W2, H2 = img2_pil.size

        output = inference(
            [tuple(images)], self.model, self.device, batch_size=1, verbose=False
        )

        # at this stage, you have the raw dust3r predictions
        pred1, pred2 = output["pred1"], output["pred2"]

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=8,
            device=self.device,
            dist="dot",
            block_size=2**13,
        )

        # Compute actual scales based on original image sizes
        long_edge1 = max(H1, W1)
        long_edge2 = max(H2, W2)
        s1 = long_edge1 / 512
        s2 = long_edge2 / 512

        matches_im0 = torch.from_numpy(matches_im0.copy()).to(self.device) * s1 + 0.5
        matches_im1 = torch.from_numpy(matches_im1.copy()).to(self.device) * s2 + 0.5

        # uniformly sample max_kpts if too many keypoints
        if matches_im0.shape[0] > max_kpts:
            indices = torch.randperm(matches_im0.shape[0])[:max_kpts]
            matches_im0 = matches_im0[indices]
            matches_im1 = matches_im1[indices]

        idx = torch.arange(matches_im0.shape[0]).repeat(2, 1).T
        return PairMatches(idx, matches_im0, matches_im1)
