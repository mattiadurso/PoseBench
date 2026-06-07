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
    def __init__(
        self,
        v2: bool = False,
        descriptor_G: bool = False,
        device: str = "cuda:0",
        border: int = 16,
    ) -> None:
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
