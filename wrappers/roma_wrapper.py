import sys
import gc
import warnings
import torch
from PIL import Image
import torch.nn.functional as F

from pathlib import Path

method_path = Path(__file__).resolve().parents[1] / "methods/RoMa"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)
from wrappers.wrapper import MethodWrapper, MethodOutput

from romatch import roma_outdoor


class RoMaWrapper(MethodWrapper):
    def __init__(
        self,
        device: str = "cuda:0",
        border=16,
    ):
        name = "roma"
        super().__init__(name=name, border=border, device=device)
        self.is_sparse_feature_extractor = False

        # Load weights
        self.model = roma_outdoor(device=device)

    def _extract(self, img1_path, img2_path, max_kpts=4096):
        """Extract keypoints and descriptors from an image."""
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

        return torch.arange(kptsA.shape[0]).repeat(2, 1).T, kptsA, kptsB
