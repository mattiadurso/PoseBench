"""Wrapper for the RDD (Robust Deep Detector/Descriptor) method."""

import sys
import yaml
import warnings
import torch

from pathlib import Path
from typing import Optional

method_path = Path(__file__).resolve().parents[1] / "methods/rdd"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)

from RDD.RDD import build
from wrappers.wrapper import MethodWrapper, MethodOutput


class RDDWrapper(MethodWrapper):
    """MethodWrapper for the RDD detector/descriptor model."""

    def __init__(
        self, device: str = "cuda:0", border: int = 16, config: Optional[dict] = None
    ) -> None:
        """Build the RDD model from its config and weights, frozen in eval mode.

        Args:
            device: Torch device to load the model on.
            border: Border (in pixels) used to discard keypoints near image edges.
            config: Optional RDD config dict; loaded from the default YAML if None.

        Raises:
            RuntimeError: If the config/weights are missing or initialization fails.
        """
        super().__init__(name="rdd", border=border, device=device)

        try:
            # Load weights
            config_path = method_path / "configs/default.yaml"
            weights_path = method_path / "weights/RDD-v1.pth"

            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            if config is None:
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)
                # print("RDD config:", config)

            self.config = config

            RDD_model = build(
                config=config,
                weights=str(weights_path),
            )
            RDD_model.eval()

            # disable gradients
            for p in RDD_model.parameters():
                p.requires_grad = False

            self.model = RDD_model.to(device)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize RDD model: {e}")

    @torch.inference_mode()
    def _extract(
        self,
        x: torch.Tensor,
        max_kpts: int = 2048,
        custom_kpts: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        """Extract keypoints, scores, and descriptors for a single image.

        Rebuilds the model if ``max_kpts`` differs from the configured ``top_k``,
        then runs RDD's extractor on the image.

        Args:
            x: Image tensor, CHW or NCHW (a leading batch dim is added if missing).
            max_kpts: Maximum number of keypoints (the model's ``top_k``).
            custom_kpts: Unused; kept for interface compatibility.

        Returns:
            MethodOutput with keypoints, keypoint scores, and descriptors.
        """
        if self.config["top_k"] != max_kpts:
            self.config["top_k"] = max_kpts
            self.__init__(device=self.device, border=self.border, config=self.config)

        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            out = self.model.extract(x)[0]
            kpts, scores, des = (
                out["keypoints"],
                out["scores"],
                out["descriptors"],
            )

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
