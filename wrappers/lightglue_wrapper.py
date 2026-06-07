import warnings
from pathlib import Path
from typing import Union

import torch


warnings.filterwarnings("ignore", category=UserWarning)
from wrappers.wrapper import MethodWrapper, PairMatches

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd


class LightGlueWrapper(MethodWrapper):
    def __init__(
        self,
        detector_name: str = "superpoint",
        device: str = "cuda:0",
        border: int = 16,
    ) -> None:
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
        self.max_kpts = None
        self.model = (
            LightGlue(features=detector_name).eval().to(device)
        )  # load the matcher

    def init_extractor(self, max_kpts: int = 2048) -> None:
        "Initialize the extractor based on the selected detector"
        self.max_kpts = max_kpts
        # Load weights
        if self.detector_name == "superpoint":
            self.detector = (
                SuperPoint(max_num_keypoints=max_kpts).eval().to(self.device)
            )
        elif self.detector_name == "disk":
            self.detector = DISK(max_num_keypoints=max_kpts).eval().to(self.device)
        elif self.detector_name == "sift":
            self.detector = SIFT(max_num_keypoints=max_kpts).eval().to(self.device)
        elif self.detector_name == "aliked":
            self.detector = ALIKED(max_num_keypoints=max_kpts).eval().to(self.device)

    def match_pair(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        max_kpts: int = 4096,
    ) -> PairMatches:
        """Match an image pair end-to-end with the detector + LightGlue."""
        if self.detector is None or self.max_kpts != max_kpts:
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

        idx = torch.arange(points1.shape[0]).repeat(2, 1).T
        return PairMatches(idx, points1 + 0.5, points2 + 0.5)

    def move_to(self, device: str = "cpu") -> "MethodWrapper":
        """Move the model to the specified device."""
        self.device = device
        self.model.to(device)
        if self.detector is not None:
            self.detector.to(device)
        return self
