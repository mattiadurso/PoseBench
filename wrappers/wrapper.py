from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Union,
    Tuple,
    List,
    Any,
    NamedTuple,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

import cv2
import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F

from common import geometry

if TYPE_CHECKING:
    from matchers.mnn import Matches

Number = Union[int, float]


@dataclass
class MethodOutput:
    kpts: Tensor
    kpts_scores: Optional[Tensor] = None
    kpts_sizes: Optional[Tensor] = None  # receptive field
    kpts_scales: Optional[Tensor] = (
        None  # at which resolution they have been extracted, for multiscale
    )
    kpts_angles: Optional[Tensor] = None
    des: Optional[Tensor] = None
    des_vol: Optional[Tensor] = None

    def __post_init__(self) -> None:
        assert self.kpts.ndim == 2, (
            f"kpts must have shape (N, 2), got {self.kpts.shape}"
        )

        # put emtpy stuff in place of None
        if self.kpts_scores is None:
            self.kpts_scores = torch.ones_like(self.kpts[:, 0])
        if self.kpts_sizes is None:
            self.kpts_sizes = torch.ones_like(self.kpts[:, 0])
        if self.kpts_scales is None:
            self.kpts_scales = torch.ones_like(self.kpts[:, 0])
        if self.kpts_angles is None:
            self.kpts_angles = torch.zeros_like(self.kpts[:, 0])

    def __getitem__(self, key: Union[str, None]) -> Tensor:
        return self.__dict__[key]

    def __contains__(self, item: object) -> bool:
        return item in self.__dict__

    def get(self, key: str) -> Optional[Any]:
        return self[key] if key in self.__dict__ else None

    def cpu(self) -> "MethodOutput":
        """Move all tensors to CPU."""
        return MethodOutput(
            kpts=self.kpts.cpu() if self.kpts is not None else None,
            kpts_scores=(
                self.kpts_scores.cpu() if self.kpts_scores is not None else None
            ),
            kpts_sizes=self.kpts_sizes.cpu() if self.kpts_sizes is not None else None,
            kpts_scales=(
                self.kpts_scales.cpu() if self.kpts_scales is not None else None
            ),
            kpts_angles=(
                self.kpts_angles.cpu() if self.kpts_angles is not None else None
            ),
            des=self.des.cpu() if self.des is not None else None,
            des_vol=self.des_vol.cpu() if self.des_vol is not None else None,
        )

    def mask(self, mask: Tensor) -> "MethodOutput":
        assert mask.dtype == torch.bool, "mask must be boolean"
        return MethodOutput(
            kpts=self.kpts.clone()[mask],
            kpts_scores=self.kpts_scores.clone()[mask],
            kpts_sizes=self.kpts_sizes.clone()[mask],
            kpts_scales=self.kpts_scales.clone()[mask],
            kpts_angles=self.kpts_angles.clone()[mask],
            des=self.des.clone()[mask] if self.des is not None else None,
            des_vol=self.des_vol.clone() if self.des_vol is not None else None,
        )


class PairMatches(NamedTuple):
    """Correspondences produced by a dense matcher for an image pair.

    Behaves like a 3-tuple ``(matches, kpts1, kpts2)`` so existing call sites that
    unpack it keep working, while exposing named, typed fields.

    Attributes:
        matches: ``(K, 2)`` integer index pairs into ``kpts1``/``kpts2``.
        kpts1: ``(K, 2)`` pixel coordinates in the first image.
        kpts2: ``(K, 2)`` pixel coordinates in the second image.
    """

    matches: Tensor
    kpts1: Tensor
    kpts2: Tensor


class MethodWrapper(ABC):
    def __init__(
        self, name: str, border: int = 0, device: str = "cpu", use_amp: bool = True
    ) -> None:
        self.name = name
        self.border = border
        self.device = device
        self.custom_descriptor = None
        self.matcher = None
        self.is_sparse_feature_extractor = True
        # amp
        self.use_amp = use_amp
        if self.use_amp:
            print("Using automatic mixed precision.")
        self.amp_dtype = torch.float16

    def load_image(self, path: Union[str, Path], scaling: float = 1.0) -> Tensor:
        """
        Load image from path, convert to float32 tensor in [0, 1], resize if needed,
        and crop to multiple of 16."""
        img = self.read_image_to_torch(str(path)) / 255.0  # 3, H, W, float32 [0, 1]
        # resize if needed
        if scaling != 1.0:
            img = F.interpolate(
                img.unsqueeze(0),
                scale_factor=1 / scaling,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        # crop to multiple of 16
        img = self.crop_to_multiple_of(img, multiple_of=16)

        return img.to(self.device)

    def read_image_to_torch(self, path: str) -> Tensor:
        """
        Read image with OpenCV and convert to RGB.
        Returns a tensor uint8 CxHxW in [0,255].
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWxC (BGR) or HxW (gray)
        if img is None:
            raise FileNotFoundError(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = torch.from_numpy(img)  # HxWxC, uint8/uint16
        t = t.permute(2, 0, 1).contiguous()  # CxHxW
        return t

    def crop_to_multiple_of(
        self, img: Union[Tensor, np.ndarray], multiple_of: int = 16
    ) -> Union[Tensor, np.ndarray]:
        if isinstance(img, np.ndarray):
            H, W = img.shape[:2]
            new_H = (H // multiple_of) * multiple_of
            new_W = (W // multiple_of) * multiple_of
            return img[:new_H, :new_W, :]

        elif isinstance(img, Tensor):
            H, W = img.shape[-2:]
            new_H = (H // multiple_of) * multiple_of
            new_W = (W // multiple_of) * multiple_of
            return img[..., :new_H, :new_W]
        else:
            raise TypeError("Unsupported image type")

    def add_custom_descriptor(
        self, model: torch.nn.Module, grad: bool = False
    ) -> None:
        # can be whatever model that takes (B, C, H, W) as input and returns (B, D, H, W)
        self.custom_descriptor = model
        if not grad:
            for p in self.custom_descriptor.parameters():
                p.requires_grad = grad
        self.custom_descriptor.to(self.device)

    def to_pixel_coords(self, flow: Tensor, h1: int, w1: int) -> Tensor:
        w_ = w1 * (flow[..., 0] + 1) / 2
        h_ = h1 * (flow[..., 1] + 1) / 2
        flow = torch.stack((w_, h_), axis=-1)
        return flow

    def _extract(
        self,
        img: Union[Tensor, np.ndarray],
        max_kpts: Union[float, int],
        custom_kpts: Optional[Tensor] = None,
    ) -> MethodOutput:
        """Extract keypoints/descriptors from a single image (sparse wrappers).

        Sparse feature extractors override this. Dense matchers override
        :meth:`match_pair` instead and never implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not a sparse feature extractor; "
            "it should implement match_pair() instead of _extract()."
        )

    def match_pair(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        max_kpts: int,
    ) -> PairMatches:
        """Match an image pair end-to-end (dense matchers like RoMa/LightGlue).

        Dense matchers override this. Sparse extractors leave it unimplemented
        (they detect per image via :meth:`extract` and match via :meth:`match`).

        Args:
            img1_path: Path to the first image.
            img2_path: Path to the second image.
            max_kpts: Maximum number of correspondences to return.

        Returns:
            A :class:`PairMatches` with index pairs and pixel coordinates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a sparse feature extractor; "
            "use extract() + match() instead of match_pair()."
        )

    @torch.inference_mode()
    def extract(
        self,
        img: Union[Tensor, np.ndarray],
        max_kpts: Union[float, int],
        custom_kpts: Optional[Tensor] = None,
    ) -> MethodOutput:
        if not isinstance(img, Tensor):
            raise TypeError("Input image must be a Tensor")

        H, W = img.shape[-2:]  # images is supposed to be (C, H, W) or (B, C, H, W)
        output = self._extract(img, max_kpts, custom_kpts)
        # ? remove all the points in the border
        valid_mask = (
            (output.kpts[:, 0] > self.border)
            & (output.kpts[:, 0] < W - self.border)
            & (output.kpts[:, 1] > self.border)
            & (output.kpts[:, 1] < H - self.border)
        )
        output = output.mask(valid_mask)
        return output

    def grid_sample_nan(
        self, xy: Tensor, img: Tensor, mode: str = "nearest"
    ) -> Tuple[Tensor, Tensor]:
        """``grid_sample`` with coordinate normalization and NaN handling.

        Thin wrapper around :func:`common.geometry.grid_sample_nan`.
        """
        return geometry.grid_sample_nan(xy, img, mode=mode)

    def normalize_pixel_coordinates(self, xy: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Normalize pixel coordinates to ``[-1, 1]``.

        Thin wrapper around :func:`common.geometry.normalize_pixel_coordinates`.
        """
        return geometry.normalize_pixel_coordinates(xy, shape)

    def match(self, des0: List[Tensor], des1: List[Tensor]) -> List["Matches"]:
        if self.matcher is None:
            raise ValueError("No matcher defined for this wrapper")
        return self.matcher.match(des0, des1)

    @staticmethod
    def _to_nchw_float01(x: torch.Tensor) -> Tuple[Tensor, bool, bool]:
        """Convert a 3D/4D image tensor to NCHW float in [0, 1].

        Returns:
            A tuple ``(x_nchw, input_was_nchw, is_batched)``.
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D/4D tensor, got {x.ndim}D with shape {tuple(x.shape)}"
            )

        # --- detect and convert to NCHW ---
        is_batched = x.ndim == 4
        if x.ndim == 3:
            if x.shape[0] in (1, 3, 4):  # CHW
                x_nchw, input_was_nchw = x.unsqueeze(0), True
            else:  # HWC
                x_nchw, input_was_nchw = x.permute(2, 0, 1).unsqueeze(0), False
        else:
            if x.shape[1] in (1, 3, 4):  # NCHW
                x_nchw, input_was_nchw = x, True
            else:  # NHWC
                x_nchw, input_was_nchw = x.permute(0, 3, 1, 2), False

        # to float32 in [0,1]
        if not torch.is_floating_point(x_nchw):
            x_nchw = x_nchw.float()
        if x_nchw.max() > 1.5:
            x_nchw = x_nchw / 255.0

        return x_nchw, input_was_nchw, is_batched

    @staticmethod
    def _to_grayscale(x_nchw: Tensor, gray_weights: Sequence[float]) -> Tensor:
        """Collapse channels to 1: luminance weights for >=3 channels, else mean."""
        if x_nchw.shape[1] >= 3:
            w = torch.tensor(
                gray_weights[:3], dtype=x_nchw.dtype, device=x_nchw.device
            ).view(1, 3, 1, 1)
            return (x_nchw[:, :3] * w).sum(dim=1, keepdim=True)  # N×1×H×W
        # already single-channel (or a weird count): average the channels
        return x_nchw.mean(dim=1, keepdim=True)

    @staticmethod
    def _resolve_mean_std(
        mean: Optional[Union[Number, Sequence[Number]]],
        std: Optional[Union[Number, Sequence[Number]]],
        C: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> Tuple[Tensor, Tensor]:
        """Build per-channel ``(mean, std)`` tensors, broadcasting scalars to C."""
        if (mean is None) ^ (std is None):
            raise ValueError("Provide both mean and std, or neither (for grayscale).")

        def to_list(v: Union[Number, Sequence[Number]]) -> List[float]:
            return [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]

        tensors = []
        for name, values in (("mean", to_list(mean)), ("std", to_list(std))):
            if len(values) == 1:
                tensors.append(torch.full((C,), values[0], dtype=dtype, device=device))
            else:
                if len(values) != C:
                    raise ValueError(f"{name} length {len(values)} != channels {C}")
                tensors.append(torch.tensor(values, dtype=dtype, device=device))
        return tensors[0], tensors[1]

    def normalize_image(
        self,
        x: torch.Tensor,
        mean: Optional[Union[Number, Sequence[Number]]] = None,
        std: Optional[Union[Number, Sequence[Number]]] = None,
        gray_weights: Sequence[float] = (
            0.2989,
            0.5870,
            0.1140,
        ),  # Y = 0.299R + 0.587G + 0.114B
    ) -> torch.Tensor:
        """
        If mean and std are None: convert RGB to grayscale.
        Else: normalize using mean/std (scalar or per-channel list).
        Preserves the input layout (HWC/CHW or NHWC/NCHW). Channel count may become 1 after grayscale.
        Expects uint8 in [0,255] or float in [0,1].
        """
        x_nchw, input_was_nchw, is_batched = self._to_nchw_float01(x)
        C = x_nchw.shape[1]

        if mean is None and std is None:
            x_nchw = self._to_grayscale(x_nchw, gray_weights)
        else:
            mean_t, std_t = self._resolve_mean_std(
                mean, std, C, x_nchw.dtype, x_nchw.device
            )
            x_nchw = (x_nchw - mean_t.view(1, C, 1, 1)) / std_t.view(1, C, 1, 1)

        # --- restore original layout ---
        if not is_batched:
            x_out = x_nchw.squeeze(0)
            return x_out if input_was_nchw else x_out.permute(1, 2, 0)
        return x_nchw if input_was_nchw else x_nchw.permute(0, 2, 3, 1)

    def move_to(self, device: str = "cpu") -> "MethodWrapper":
        """Move the model to the specified device."""
        self.device = device
        self.model.to(device)
        if self.custom_descriptor is not None:
            self.custom_descriptor.to(device)
        return self

    # retrocompatibility
    def img_from_numpy(self, img: np.ndarray, device: str = None) -> Tensor:
        device = device if device is not None else self.device

        s = 14 if self.name.endswith("-G") else 16

        img = img[: img.shape[0] // s * s, : img.shape[1] // s * s]
        img_out = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.0
        return img_out.to(self.device).half()
