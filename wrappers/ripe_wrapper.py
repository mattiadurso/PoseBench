"""Wrapper for the RIPE keypoint detector and descriptor."""

import sys
import torch
from pathlib import Path
from typing import Optional

sys.path.append("methods/ripe")
from methods.ripe.ripe import vgg_hyper
from wrappers.wrapper import MethodWrapper, MethodOutput


class RIPEWrapper(MethodWrapper):
    """MethodWrapper adapter around the RIPE model."""

    def __init__(self, device: str = "cuda", border: int = 16) -> None:
        """Initialize the RIPE wrapper.

        Args:
            device (str): Torch device to load the model on.
            border (int): Border margin used to discard keypoints near image edges.
        """
        super().__init__(name="ripe", border=border, device=device)

        model_path = "methods/ripe/ckpt/ripe_weights.pth"
        self.model = vgg_hyper(Path(model_path)).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _extract(
        self,
        x: torch.Tensor,
        max_kpts: int = 2048,
        custom_kpts: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        """Extract keypoints and descriptors from an image.

        Keypoints are returned with a +0.5 pixel offset. If a custom descriptor is
        set, descriptors are resampled from its feature volume at the detected
        keypoints instead of using RIPE's own descriptors.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
            max_kpts (int): Maximum number of keypoints to extract.
            custom_kpts (Tensor, optional): Custom keypoints; not supported and raises
                NotImplementedError if provided.

        Returns:
            MethodOutput: Contains keypoints (with +0.5 offset), scores, and descriptors.
        """
        x = x if x.dim() == 4 else x[None]

        if custom_kpts is not None:
            raise NotImplementedError("Custom keypoints not implemented for RIPE.")

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            kpts, des, scores = self.model.detectAndCompute(
                x, threshold=0.5, top_k=max_kpts
            )

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(x)
                des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T

        return MethodOutput(kpts=kpts + 0.5, kpts_scores=scores, des=des)
