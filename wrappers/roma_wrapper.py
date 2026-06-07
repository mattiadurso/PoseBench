"""Wrapper for the RoMa dense feature matcher."""

import sys
import warnings
import torch
from PIL import Image

from pathlib import Path
from typing import Union

method_path = Path(__file__).resolve().parents[1] / "methods/RoMa"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)
from wrappers.wrapper import MethodWrapper, PairMatches

from romatch import roma_outdoor


class RoMaWrapper(MethodWrapper):
    """Wraps the pretrained outdoor RoMa model as a dense pair matcher."""

    def __init__(
        self,
        device: str = "cuda:0",
        border: int = 16,
    ) -> None:
        """Load the pretrained outdoor RoMa model.

        Args:
            device: Torch device to load the model onto.
            border: Image border margin (pixels) inherited by MethodWrapper.
        """
        name = "roma"
        super().__init__(name=name, border=border, device=device)
        self.is_sparse_feature_extractor = False

        # Load weights
        self.model = roma_outdoor(device=device)

    def match_pair(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        max_kpts: int = 4096,
    ) -> PairMatches:
        """Match an image pair end-to-end with RoMa."""
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        W1, H1 = img1.size
        W2, H2 = img2.size

        # Match
        warp, certainty = self.model.match(img1_path, img2_path, device=self.device)
        # Sample matches for estimation
        matches, certainty = self.model.sample(warp, certainty)
        # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
        kptsA, kptsB = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        # uniformly sample max_kpts if too many keypoints
        if kptsA.shape[0] > max_kpts:
            indices = torch.randperm(kptsA.shape[0])[:max_kpts]
            kptsA = kptsA[indices]
            kptsB = kptsB[indices]

        idx = torch.arange(kptsA.shape[0]).repeat(2, 1).T
        return PairMatches(idx, kptsA + 0.5, kptsB + 0.5)
