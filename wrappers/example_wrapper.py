"""Non-working template wrapper showing how to adapt a new method."""

# NOT working class

import sys
import torch

sys.path.append("methods/method")

from methods.method import method  # noqa: F401  (template: shows how to import a method)
from wrappers.wrapper import MethodWrapper, MethodOutput


class ExampleWrapper(MethodWrapper):
    """Non-working template MethodWrapper to copy when adding a new method."""

    def __init__(self, device: str = "cuda", border: int = 16) -> None:
        """Initialize the template wrapper (model is left as None placeholder).

        Args:
            device (str): Torch device to load the model on.
            border (int): Border margin used to discard keypoints near image edges.
        """
        super().__init__(name="aliked", border=border, device=device)

        self.model = None  # .eval().to(device)

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor, max_kpts: int = 2048) -> MethodOutput:
        """Extract keypoints and descriptors (template; not functional).

        Args:
            x (Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
            max_kpts (int): Maximum number of keypoints to keep.

        Returns:
            MethodOutput: Contains keypoints, scores, and descriptors.
        """
        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            out = self.model(x)

        kpts = out["kpts"][:max_kpts]
        scores = out["scores"][:max_kpts]
        des = out["des"][:max_kpts]

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
