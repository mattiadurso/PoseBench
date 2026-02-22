import sys
import gc
import warnings
import torch
from PIL import Image
import torch.nn.functional as F

from pathlib import Path

method_path = Path(__file__).resolve().parents[1] / "methods/RoMa"
sys.path.append(str(method_path))
from wrappers.wrapper import MethodWrapper, MethodOutput

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import kornia as K
import kornia.feature as KF


class LoFTRWrapper(MethodWrapper):
    def __init__(
        self,
        device: str = "cuda:0",
        border=16,
    ):
        name = "loftr"
        super().__init__(name=name, border=border, device=device)
        self.is_sparse_feature_extractor = False

        # Load weights
        self.model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _extract(self, img1_path, img2_path, max_kpts=4096):
        """Extract keypoints and descriptors from an image."""
        img1 = K.io.load_image(img1_path, K.io.ImageLoadType.RGB32).to(self.device)
        img2 = K.io.load_image(img2_path, K.io.ImageLoadType.RGB32).to(self.device)

        scale1, img1 = self.resize_image(img1, long_edge_target=600)
        scale2, img2 = self.resize_image(img2, long_edge_target=600)

        # Match
        input_dict = {
            # LofTR works on grayscale images only
            "image0": self.to_grayscale(img1)[None],
            "image1": self.to_grayscale(img2)[None],
        }

        with torch.inference_mode():
            correspondences = self.model(input_dict)

        mkpts0 = correspondences["keypoints0"] / scale1
        mkpts1 = correspondences["keypoints1"] / scale2

        # uniformly sample max_kpts if too many keypoints
        if mkpts0.shape[0] > max_kpts:
            indices = torch.randperm(mkpts0.shape[0])[:max_kpts]
            mkpts0 = mkpts0[indices]
            mkpts1 = mkpts1[indices]

        return torch.arange(mkpts0.shape[0]).repeat(2, 1).T, mkpts0 + 0.5, mkpts1 + 0.5

    def to_grayscale(self, img):
        """Convert image to grayscale if needed."""
        if img.shape[0] == 3:
            img = K.color.rgb_to_grayscale(img)
        return img

    def resize_image(self, image, long_edge_target=600):
        """Get the shape after border reduction."""
        H, W = image.shape[-2:]
        scale = long_edge_target / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        image_resized = K.geometry.resize(image, (new_H, new_W), antialias=True)
        return scale, image_resized
