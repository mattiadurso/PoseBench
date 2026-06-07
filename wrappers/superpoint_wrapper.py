"""Wrapper for the SuperPoint keypoint detector and descriptor."""

from __future__ import annotations

import sys
import torch

sys.path.append("methods/superpoint")

from wrappers.wrapper import MethodWrapper, MethodOutput
from methods.superpoint.models.superpoint import SuperPoint


class SuperPointWrapper(MethodWrapper):
    """MethodWrapper adapter around the SuperPoint model."""

    def __init__(self, device: str, border: int = 16) -> None:
        """Initialize the SuperPoint wrapper.

        Args:
            device (str): Torch device to load the model on.
            border (int): Border margin used to discard keypoints near image edges;
                also used to set the reported keypoint sizes.
        """
        super().__init__(name="SuperPoint", border=border, device=device)
        config = {
            "keypoint_threshold": -1,  # min score, -1 to disable
            "max_keypoints": 2048,
        }
        self.model = SuperPoint(config).to(device)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def _extract(
        self,
        img: torch.Tensor,
        max_kpts: float | int,
        custom_kpts: torch.Tensor | None = None,
    ) -> MethodOutput:
        """Extract keypoints and descriptors from an image.

        Keypoints are sorted by score, truncated to max_kpts, and returned with a
        +0.5 pixel offset. If a custom descriptor is set, descriptors are resampled
        from its feature volume instead of using SuperPoint's own descriptors.

        Args:
            img (Tensor): Input image tensor of shape (C, H, W); must not be batched.
            max_kpts (float | int): Maximum number of keypoints to keep; updates the
                model config if it differs from the current value.
            custom_kpts (Tensor, optional): Custom keypoints; not supported and raises
                NotImplementedError if provided.

        Returns:
            MethodOutput: Keypoints (with +0.5 offset), scores, sizes, and descriptors.
        """
        if max_kpts != self.model.config["max_keypoints"]:
            self.model.config["max_keypoints"] = max_kpts
            print(f"Updated max_keypoints to {max_kpts}.")

        assert img.ndim == 3, "image must be not batched"

        if custom_kpts is not None:
            raise NotImplementedError(
                "Custom keypoints not implemented for SuperPoint."
            )

        with torch.amp.autocast(self.device, enabled=False):
            output = self.model({"image": self.normalize_image(img)[None]})

            kpts = output["keypoints"][0]
            kpts_scores = output["scores"][0]

            idxs = kpts_scores.argsort(descending=True)
            idxs = idxs[: min(idxs.shape[0], max_kpts)]

            kpts = kpts[idxs] + 0.5
            kpts_scores = kpts_scores[idxs]
            kpts_sizes = (2 * self.border) * torch.ones_like(idxs)

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(img[None])
                descriptors = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[
                    0
                ][0].T
            else:
                descriptors = output["descriptors"][0].permute(1, 0)  # N,256
                descriptors = descriptors[idxs]

        output = MethodOutput(
            kpts=kpts, kpts_scores=kpts_scores, kpts_sizes=kpts_sizes, des=descriptors
        )

        return output
