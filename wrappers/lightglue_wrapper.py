import sys
import gc
import warnings
import torch
from PIL import Image
import torch.nn.functional as F

from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
from wrappers.wrapper import MethodWrapper, MethodOutput

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd


class LightGlueWrapper(MethodWrapper):
    def __init__(
        self,
        detector_name: str = "superpoint",
        device: str = "cuda:0",
        border=16,
    ):
        assert detector_name in [
            "superpoint",
            "disk",
            "sift",
            "aliked",
        ], "Detector not supported for LightGlueWrapper."
        name = detector_name + "lightglue"
        super().__init__(name=name, border=border, device=device)
        self.is_sparse_feature_extractor = False

        self.detector_name = detector_name
        self.detector = None
        self.model = LightGlue(features=detector_name).eval().cuda()  # load the matcher

    def init_extractor(self, max_kpts=2048):
        "Initialize the extractor based on the selected detector"
        # Load weights
        if self.detector_name == "superpoint":
            self.detector = (
                SuperPoint(max_num_keypoints=max_kpts).eval().cuda()
            )  # load the extractor
        elif self.detector_name == "disk":
            self.detector = (
                DISK(max_num_keypoints=max_kpts).eval().cuda()
            )  # load the extractor
        elif self.detector_name == "sift":
            self.detector = (
                SIFT(max_num_keypoints=max_kpts).eval().cuda()
            )  # load the extractor
        elif self.detector_name == "aliked":
            self.detector = (
                ALIKED(max_num_keypoints=max_kpts).eval().cuda()
            )  # load the extractor

    def _extract(self, img1_path, img2_path, max_kpts=4096):
        """Extract keypoints and descriptors from an image."""
        if self.detector is None:
            self.init_extractor(max_kpts)

        img1 = load_image(img1_path).to(self.device)
        img2 = load_image(img2_path).to(self.device)
        # extract local features
        feats1 = self.detector.extract(
            img1
        )  # auto-resize the image, disable with resize=None
        feats2 = self.detector.extract(img2)

        # match the features
        matches01 = self.model({"image0": feats1, "image1": feats2})
        feats1, feats2, matches01 = [
            rbd(x) for x in [feats1, feats2, matches01]
        ]  # remove batch dimension
        matches = matches01["matches"]  # indices with shape (K,2)
        points1 = feats1["keypoints"][
            matches[..., 0]
        ]  # coordinates in image #0, shape (K,2)
        points2 = feats2["keypoints"][
            matches[..., 1]
        ]  # coordinates in image #1, shape (K,2)

        return torch.arange(points1.shape[0]).repeat(2, 1).T, points1, points2

    def move_to(self, device="cpu"):
        """Move the model to the specified device."""
        self.device = device
        self.model.to(device)
        if self.detector is not None:
            self.detector.to(device)
        return self
