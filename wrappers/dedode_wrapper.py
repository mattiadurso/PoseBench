"""Wrapper for the DeDoDe detector/descriptor method."""

import sys
import gc
import warnings
import torch
import torch.nn.functional as F

from pathlib import Path
from typing import Optional

method_path = Path(__file__).resolve().parents[1] / "methods/dedode"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)

from methods.dedode.DeDoDe import (
    dedode_detector_L,
    dedode_descriptor_B,
    dedode_descriptor_G,
)
from wrappers.wrapper import MethodWrapper, MethodOutput


class DeDoDeWrapper(MethodWrapper):
    """MethodWrapper for the DeDoDe detector and descriptor models."""

    def __init__(
        self,
        v2: bool = False,
        descriptor_G: bool = False,
        device: str = "cuda:0",
        border: int = 16,
    ) -> None:
        """Load the DeDoDe detector and chosen descriptor with their weights.

        Args:
            v2: Use the v2 detector weights instead of the v1 detector.
            descriptor_G: Use the G descriptor instead of the default B descriptor.
            device: Torch device to load the models on.
            border: Border (in pixels) used to discard keypoints near image edges.
        """
        name = "dedode" if not v2 else "dedode2"
        name += "-G" if descriptor_G else "-B"
        super().__init__(name=name, border=border, device=device)

        # Load weights
        if v2:
            detector_path = method_path / "weights/dedode_detector_L_v2.pth"
        else:
            detector_path = method_path / "weights/dedode_detector_L.pth"
        descriptor_G_path = method_path / "weights/dedode_descriptor_G.pth"
        descriptor_B_path = method_path / "weights/dedode_descriptor_B.pth"

        self.detector = dedode_detector_L(
            weights=torch.load(detector_path, map_location=device), device=device
        )
        self.descriptor_G = descriptor_G
        if descriptor_G:
            self.descriptor = dedode_descriptor_G(
                weights=torch.load(descriptor_G_path, map_location=device),
                device=device,
            )
        else:
            self.descriptor = dedode_descriptor_B(
                weights=torch.load(descriptor_B_path, map_location=device),
                device=device,
            )

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def add_custom_descriptor(
        self, model: torch.nn.Module, grad: bool = False
    ) -> None:
        """Replace the DeDoDe descriptor with a custom descriptor network.

        Moves the model to the wrapper device, frees the original descriptor, and
        runs garbage collection / empties the CUDA cache.

        Args:
            model: Custom descriptor network used in place of the DeDoDe descriptor.
            grad: If False, freeze the custom descriptor's parameters.
        """
        self.custom_descriptor = model
        if not grad:
            for p in self.custom_descriptor.parameters():
                p.requires_grad = grad
        self.custom_descriptor.to(self.device)
        # clean up
        self.descriptor = None
        self.descriptor_G = False
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _extract(
        self,
        x: torch.Tensor,
        max_kpts: int = 2048,
        custom_kpts: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        """Detect keypoints and compute descriptors for a single image.

        Normalizes the image with ImageNet stats, detects keypoints (or uses
        provided custom keypoints), and describes them with the DeDoDe descriptor
        (L2-normalized) or a custom descriptor sampled at keypoint locations. The
        G descriptor path crops the image to a multiple of 14 first.

        Args:
            x: Image tensor, CHW or NCHW (a leading batch dim is added if missing).
            max_kpts: Maximum number of keypoints to detect.
            custom_kpts: Optional pixel-coordinate keypoints to describe instead of
                running the detector; their scores are set to ones.

        Returns:
            MethodOutput with pixel keypoints, keypoint scores, and descriptors.
        """
        x = x if x.dim() == 4 else x[None]

        # eventually cropping/padding to multiples of 14
        if self.descriptor_G:
            x = self.crop_to_multiple_of(x, multiple_of=14)

        batch = {"image": self.normalize_image(x, self.mean, self.std)}

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # detector
            if custom_kpts is None:
                out = self.detector.detect(batch, num_keypoints=max_kpts)
                kpts, scores = out["keypoints"], out["confidence"]
                kpts_pix = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
            else:  # use the given custom kpts
                kpts_pix = custom_kpts.to(x.device)
                scores = torch.ones(kpts_pix.shape[0], device=x.device)

            # descriptors
            if self.custom_descriptor is None:
                out = self.descriptor.describe_keypoints(batch, kpts)
                des = out["descriptions"][0]
                # L2 normalization, needed since not done in dedode descriptor
                des = F.normalize(des, p=2, dim=-1)

            else:  # custom descriptor network
                des_vol = self.custom_descriptor(x)
                des = self.grid_sample_nan(kpts_pix[None], des_vol, mode="nearest")[0][
                    0
                ].permute(1, 2, 0)[0]

        return MethodOutput(kpts=kpts_pix[0], kpts_scores=scores[0], des=des)

    def move_to(self, device: str = "cpu") -> "MethodWrapper":
        """Move the model to the specified device."""
        self.device = device
        self.detector.to(device)
        if self.custom_descriptor is not None:
            self.custom_descriptor.to(device)
        else:
            self.descriptor.to(device)
        return self
