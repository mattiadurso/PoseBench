"""Wrapper for kornia's ALIKED implementation.

Distinct from :mod:`wrappers.aliked_wrapper`, which drives the vendored copy of
the original ALIKED repo. Both load the *same* weight files (verified byte
identical, MD5 8411636df70dbb476c0746a77802b33b for aliked-n16), but kornia
fetches them from its own mirror rather than the vendored `methods/aliked`
tree, and the two implementations differ in sub-pixel handling: kornia maps
normalised coordinates with ``(w - 1)`` while the vendored wrapper's
``to_pixel_coords`` uses ``w``, a ~0.5 px offset.

kornia's ALIKED returns keypoints already in pixel coordinates, so no
conversion is applied here -- doing so would double-convert.

Requires kornia >= 0.8.3 (ALIKED was added in that release), which needs
Python >= 3.11; use the `sandesc_kornia` env.
"""

from typing import Optional

import torch

from wrappers.wrapper import MethodOutput, MethodWrapper


class ALIKEDKorniaWrapper(MethodWrapper):
    """MethodWrapper around ``kornia.feature.ALIKED``."""

    def __init__(
        self,
        device: str = "cuda",
        max_kpts: int = 2048,
        border: int = 16,
        model_name: str = "aliked-n16",
    ) -> None:
        """Load kornia's ALIKED with weights from the kornia mirror.

        Args:
            device: Torch device to load the model on.
            max_kpts: Maximum number of keypoints. kornia takes this at
                construction as ``max_num_keypoints``; it is re-read on every
                _extract call, so a changed budget rebuilds the model (as the
                vendored wrapper does).
            border: Border margin used to discard keypoints near image edges.
            model_name: One of 'aliked-t16', 'aliked-n16', 'aliked-n16rot',
                'aliked-n32'. Defaults to 'aliked-n16', the variant published
                ALIKED numbers use (the vendored wrapper hardcodes
                'aliked-n16rot').
        """
        super().__init__(name="aliked-kornia", border=border, device=device)
        from kornia.feature import ALIKED

        self._ALIKED = ALIKED
        self.model_name = model_name
        self.max_kpts = max_kpts
        self.model = (
            ALIKED.from_pretrained(
                model_name=model_name,
                max_num_keypoints=max_kpts,
                device=torch.device(device),
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
            max_kpts: Maximum number of keypoints; rebuilds the model if it
                differs from the budget the model was constructed with.
            custom_kpts: Optional ``(N, 2)`` pixel keypoints to describe
                instead of detecting. Scores are set to ones, matching the
                vendored wrapper's behaviour (keypoint scores are unused
                downstream).

        Returns:
            MethodOutput with pixel keypoints, scores and descriptors.
        """
        if self.max_kpts != max_kpts:
            custom_descriptor = self.custom_descriptor
            self.__init__(
                device=self.device,
                max_kpts=max_kpts,
                border=self.border,
                model_name=self.model_name,
            )
            self.custom_descriptor = custom_descriptor

        x = x if x.dim() == 4 else x[None]

        if self.custom_descriptor is None:
            feats = self.model(x)[0]
            return MethodOutput(
                kpts=feats.keypoints,
                kpts_scores=feats.keypoint_scores,
                des=feats.descriptors,
            )

        # Custom-descriptor path. kornia's ALIKED computes its own descriptors
        # inside forward() and offers no way to skip them, so they are computed
        # and discarded here; the cost is small next to detection.
        if custom_kpts is None:
            feats = self.model(x)[0]
            kpts, scores = feats.keypoints, feats.keypoint_scores
        else:
            kpts = custom_kpts.to(x.device)
            scores = torch.ones(kpts.shape[0], device=x.device)

        des_vol = self.custom_descriptor(x)
        des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T
        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
