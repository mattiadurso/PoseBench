"""Wrappers for the DISK keypoint detector and descriptor (Kornia and original)."""

import sys
import torch
import kornia.feature as KF

from typing import Optional

from wrappers.wrapper import MethodWrapper, MethodOutput


class DiskWrapperKornia(MethodWrapper):
    """MethodWrapper adapter around Kornia's DISK implementation."""

    def __init__(self, device: str = "cuda:0", border: int = 16) -> None:
        """Initialize the Kornia DISK wrapper.

        Args:
            device (str): Torch device to load the model on.
            border (int): Border margin used to discard keypoints near image edges.
        """
        super().__init__(name="disk", border=border, device=device)

        self.model = KF.DISK.from_pretrained("depth").to(device)

        # params no grad
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def _extract(
        self,
        img: torch.Tensor,
        max_kpts: int,
        custom_kpts: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        """Extract keypoints and descriptors from an image.

        Keypoints are returned with a +0.5 pixel offset. If a custom descriptor is
        set, descriptors are resampled from its feature volume instead of using DISK's.

        Args:
            img (Tensor): Input image tensor of shape (C, H, W).
            max_kpts (int): Maximum number of keypoints to extract.
            custom_kpts (Tensor, optional): Custom keypoints; currently ignored.

        Returns:
            MethodOutput: Contains keypoints (with +0.5 offset) and descriptors.
        """
        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            out = self.model(img[None], max_kpts)[0]
            kpts, des = out.keypoints, out.descriptors

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(img[None])
                des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T

        return MethodOutput(kpts=kpts + 0.5, des=des)


try:
    # If original implemenatition is available, one can use it
    sys.path.append("methods/disk")
    from methods.disk.disk import DISK

    class DiskWrapper(MethodWrapper):
        """MethodWrapper adapter around the original DISK implementation."""

        def __init__(self, device: str = "cuda:0", border: int = 16) -> None:
            """Initialize the original DISK wrapper from local weights.

            Args:
                device (str): Torch device to load the model on.
                border (int): Border margin used to discard keypoints near image edges.
            """
            super().__init__(name="disk", border=border, device=device)
            weights_path = "methods/disk/depth-save.pth"

            disk = DISK(window=8, desc_dim=128)
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            disk.load_state_dict(state_dict["extractor"])

            self.model = disk.to(device)

        @torch.inference_mode()
        def _extract(
            self,
            img: torch.Tensor,
            max_kpts: int,
            custom_kpts: Optional[torch.Tensor] = None,
        ) -> MethodOutput:
            """Extract keypoints and descriptors using NMS detection.

            Keypoints and descriptors are sorted by descending score and keypoints
            are returned with a +0.5 pixel offset.

            Args:
                img (Tensor): Input image tensor of shape (C, H, W).
                max_kpts (int): Maximum number of keypoints to extract.
                custom_kpts (Tensor, optional): Custom keypoints; currently a no-op.

            Returns:
                MethodOutput: Keypoints (with +0.5 offset), scores, and descriptors.
            """
            with torch.amp.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                if custom_kpts is not None:
                    pass
                # desc_vol is None if use_disk_descriptors is False
                features = self.model.features(
                    img[None], kind="nms", window_size=5, cutoff=0, n=max_kpts
                )
                kpts, kpts_scores = features[0].kp, features[0].kp_logp
                des = features[0].desc

                # ? order keypoints and descriptors by scores
                order = kpts_scores.argsort(descending=True)
                kpts_scores = kpts_scores[order]
                kpts = kpts[order]
                des = des[order]

            return MethodOutput(kpts=kpts + 0.5, kpts_scores=kpts_scores, des=des)

except ImportError:
    print(
        "Could not import DISK from methods/disk. Download original implementation to enable it."
    )
