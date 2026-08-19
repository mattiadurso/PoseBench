"""Wrapper for kornia's DeDoDe implementation.

Distinct from :mod:`wrappers.dedode_wrapper`, which drives the vendored DeDoDe
repo and loads weights from `methods/dedode/weights`. This one pulls weights
from kornia's mirror via ``DeDoDe.from_pretrained``.

kornia's ``DeDoDe.forward`` returns keypoints already denormalised to pixel
coordinates, so no conversion is applied here.

Requires the `sandesc_kornia` env (kornia >= 0.8.3, Python >= 3.11).
"""

from typing import Optional

import torch

from wrappers.wrapper import MethodOutput, MethodWrapper


class DeDoDeKorniaWrapper(MethodWrapper):
    """MethodWrapper around ``kornia.feature.DeDoDe``."""

    def __init__(
        self,
        device: str = "cuda",
        border: int = 16,
        detector_weights: str = "L-C4-v2",
        descriptor_weights: str = "G-upright",
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Load kornia's DeDoDe with weights from the kornia mirror.

        Args:
            device: Torch device to load the model on.
            border: Border margin used to discard keypoints near image edges.
            detector_weights: Detector variant ('L-upright', 'L-C4', 'L-SO2',
                'L-C4-v2'). kornia's default is 'L-C4-v2'.
            descriptor_weights: Descriptor variant ('B-upright', 'G-upright',
                'B-C4', ...). kornia's default is 'G-upright'.
            amp_dtype: torch.float16 on CUDA, torch.float32 on CPU/MPS.
        """
        name = f"dedode-kornia-{descriptor_weights.split('-')[0]}"
        super().__init__(name=name, border=border, device=device)
        from kornia.feature import DeDoDe

        self.model = (
            DeDoDe.from_pretrained(
                detector_weights=detector_weights,
                descriptor_weights=descriptor_weights,
                amp_dtype=amp_dtype,
            )
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def _extract(
        self,
        x: torch.Tensor,
        max_kpts: int = 2048,
        custom_kpts: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        """Detect keypoints and describe them, or describe given keypoints.

        Args:
            x: Image tensor, ``(C, H, W)`` or ``(B, C, H, W)``.
            max_kpts: Number of keypoints to detect.
            custom_kpts: Optional ``(N, 2)`` pixel keypoints to describe
                instead of detecting; their scores are set to ones.

        Returns:
            MethodOutput with pixel keypoints, scores and descriptors.
        """
        x = x if x.dim() == 4 else x[None]

        if self.custom_descriptor is None:
            kpts, scores, des = self.model(x, n=max_kpts)
            return MethodOutput(kpts=kpts[0], kpts_scores=scores[0], des=des[0])

        # Custom-descriptor path: DeDoDe describes inside forward() with no way
        # to skip it, so the descriptors are computed and discarded.
        if custom_kpts is None:
            kpts, scores, _des = self.model(x, n=max_kpts)
            kpts, scores = kpts[0], scores[0]
        else:
            kpts = custom_kpts.to(x.device)
            scores = torch.ones(kpts.shape[0], device=x.device)

        des_vol = self.custom_descriptor(x)
        des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T
        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
