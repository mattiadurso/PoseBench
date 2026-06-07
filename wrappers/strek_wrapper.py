from __future__ import annotations

from pathlib import Path
import sys

file_path = Path("/home/mattia/Desktop/Repos/strek")
sys.path.append(str(file_path))

import cv2
import numpy as np
import torch as th
from torch import Tensor
from torchvision import transforms
import torch
from torch.nn import functional as F

from libutils.utils_2D import extract_maxima_from_map, grid_sample_nan
from libutils.utils_image import crop_multiple_of
from libutils.utils_matches import Matcher
from wrappers.Strek.network_detector import eCNN
from wrappers.Strek.network_descriptor_original import Unet
from wrappers.Strek.serial_sampling import SerialSampler
from utils.method_wrapper import MethodWrapper, MethodOutput


class StrekWrapper(MethodWrapper):
    def __init__(
        self,
        model: object | None = None,
        border: int = 16,
        multiscale: list = [1.0, 0.707, 0.5, 0.354, 0.25],
        inference_det_map_thr: float = 0.0,
        matcher: Matcher | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__("S-TREK", border=border, device=device)

        self.is_sparse_feature_extractor = True
        self.inference_det_map_thr = inference_det_map_thr
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        detector_model_path = file_path / "pretrained_models/detector_paper.pt"
        descriptor_model_path = file_path / "pretrained_models/descriptor_paper.pt"

        if model is None:
            detector_state_dict = th.load(
                detector_model_path, map_location="cpu", weights_only=True
            )
            model = eCNN(
                c_in=1, border=border, receptive_field=29, normalization="batchnorm"
            ).to(device)
            model.load_state_dict(detector_state_dict)
            model.requires_grad_(False)
            model.eval()

        self.model = model
        self.softmax_temperature = 0.01
        self.sampler_radius = 6

        self.thr_detector = inference_det_map_thr
        self.multiscale = multiscale
        self.matcher = matcher

        if descriptor_model_path is not None:
            descriptor_network = Unet(ch_in=1, norm="batch", activ="relu").to(device)
            descriptor_state_dict = th.load(
                descriptor_model_path, map_location=device, weights_only=True
            )
            descriptor_network.load_state_dict(descriptor_state_dict)
            descriptor_network.requires_grad_(False)
            descriptor_network.eval()
            self.descriptor_network = descriptor_network
        else:
            self.descriptor_network = None

        self.img_dimension_multiple_of = 8

    def img_from_numpy(self, img: np.ndarray, device: str = None) -> Tensor:
        device = device if device is not None else self.device
        # print('WARNING: forcing image resolution multiple of 16')
        img = img[: img.shape[0] // 16 * 16, : img.shape[1] // 16 * 16]
        img_out = self.transform(img.copy()).to(device)
        return img_out

    def add_descriptors_to_output(
        self, img: Tensor, output: MethodOutput
    ) -> MethodOutput:
        # des_vol, conf = self.descriptor_network(img[None])
        des_vol = self.descriptor_network(img[None])
        conf = None
        des, _ = grid_sample_nan(
            output.kpts[None], des_vol, mode="nearest"
        )  # B,des_dim,max_n_keypoints
        output.des = des[0].T
        output.des_vol = des_vol

        if conf is not None:
            des_scores, _ = grid_sample_nan(
                output.kpts[None], conf[:, None, :, :], mode="nearest"
            )  # B,1,max_n_keypoints
            des_scores = des_scores[0, 0]
        else:
            des_scores = th.ones_like(output.des[:, 0])
        output.des_scores = des_scores
        return output

    def _extract_upscale(self, img: Tensor, max_kpts: int) -> MethodOutput:
        scale = 2.0
        img_scaled = th.nn.functional.interpolate(
            img[None], scale_factor=scale, mode="bilinear", align_corners=False
        )[0]
        det_map_norm = self.compute_detmap(img_scaled)
        output = self.get_keypoints_from_detmap(
            det_map_norm, max_kpts, self.thr_detector
        )
        output.kpts /= scale
        output.kpts_sizes /= scale
        output.kpts_scales = scale * th.ones_like(output.kpts_sizes)

        # ? eventually extract descriptors
        if self.descriptor_network is not None:
            output = self.add_descriptors_to_output(img, output)

        return output

    def _extract_multiscale(self, img: Tensor, max_kpts: int) -> MethodOutput:
        all_kpts = th.zeros((0, 2), device=img.device)
        all_scores = th.zeros((0,), device=img.device)
        all_sizes = th.zeros((0,), device=img.device)
        all_scales = th.zeros((0,), device=img.device)
        all_des = th.zeros((0, 128), device=img.device)
        all_des_scores = th.zeros((0,), device=img.device)

        max_scale_policy = []
        # # ? just for plotting purposes
        # det_maps = []
        # images = []

        for i, scale in enumerate(self.multiscale):
            img_scaled = th.nn.functional.interpolate(
                img[None], scale_factor=scale, mode="bilinear", align_corners=False
            )[0]
            if min(img_scaled.shape[1:]) < 128:
                continue

            IS_USING_UNET = True
            if IS_USING_UNET:
                # print('WARNING: forcing image resolution multiple of 16')
                img_scaled = crop_multiple_of(img_scaled, 16)

            if i == 0:
                det_map_norm = self.compute_detmap(img_scaled, max_scale_policy)
            else:
                det_map_norm = self.compute_detmap(img_scaled)
            output = self.get_keypoints_from_detmap(
                det_map_norm, max_kpts, self.thr_detector
            )

            # ? eventually extract descriptors
            if self.descriptor_network is not None:
                output = self.add_descriptors_to_output(img_scaled, output)
                all_des = th.cat((all_des, output.des), dim=0)
                all_des_scores = th.cat((all_des_scores, output.des_scores), dim=0)

            kpts = output.kpts / scale
            scores = output.kpts_scores
            sizes = output.kpts_sizes / scale
            scales = scale * th.ones_like(sizes)

            all_kpts = th.cat((all_kpts, kpts), dim=0)
            all_scores = th.cat((all_scores, scores), dim=0)
            all_sizes = th.cat((all_sizes, sizes), dim=0)
            all_scales = th.cat((all_scales, scales), dim=0)

            # # ? just for plotting purposes
            # images.append(img_scaled)
            # det_maps.append(det_map_norm)

        # ? sort evetything by keypoints score
        sort_idx = th.argsort(all_scores, descending=True)[:max_kpts]
        output = MethodOutput(kpts=all_kpts[sort_idx], kpts_scores=all_scores[sort_idx])
        output.kpts_sizes = all_sizes[sort_idx]
        output.kpts_scales = all_scales[sort_idx]
        output.kpts_angles = th.zeros_like(output.kpts_sizes)
        output.des = all_des[sort_idx] if all_des.shape[0] > 0 else None
        output.des_scores = (
            all_des_scores[sort_idx] if all_des_scores.shape[0] > 0 else None
        )
        # output.policy = max_scale_policy[0]

        return output

    @th.no_grad()
    def _extract(
        self, img: Tensor, max_kpts: int, embs: Tensor | None = None
    ) -> MethodOutput:
        if self.multiscale:
            return self._extract_multiscale(img, max_kpts)

        policy = []
        det_map_norm = self.compute_detmap(img, policy)
        output = self.get_keypoints_from_detmap(
            det_map_norm, max_kpts, self.thr_detector
        )
        output.policy = policy[0]

        # ? eventually extract descriptors
        if self.descriptor_network is not None:
            output = self.add_descriptors_to_output(img, output)

        if self.descriptor_network is not None:
            output.full_des_vol = self.descriptor_network(img[None])

        return output

    @th.no_grad()
    def _extract_with_additional_descriptors_from_keypoints(
        self, img: Tensor | np.ndarray, max_kpts: float | int, kpts_additional: Tensor
    ) -> tuple[MethodOutput, Tensor]:
        output = self._extract(img, max_kpts)
        # ? eventually extract descriptors
        des_vol, conf = self.descriptor_network(img[None])
        des, _ = grid_sample_nan(
            kpts_additional[None], des_vol, mode="nearest"
        )  # B,des_dim,max_n_keypoints
        des = des[0].permute(1, 0)  # max_n_keypoints,des_dim
        return output, des

    def compute_detmap(self, img: Tensor, policy_output: list = None) -> Tensor:
        assert img.ndim == 3, "image must be not batched"
        img = img.clone()
        # ? the crop keep the top-left pixel such that the coordinated do not change
        img = crop_multiple_of(img[None], self.img_dimension_multiple_of)  # 1,1,H0,W0

        # start, end = timer_start()
        logits = self.model(img)  # 1xHxW
        # timer_end(start, end, 'model')

        policy = SerialSampler(
            logits,
            softmax_temperature=self.softmax_temperature,
            radius=self.sampler_radius,
            min_probability_mass_for_sampling=0.0,
        )  # 1,H,W

        det_map = policy.probs
        det_map_norm = det_map / det_map.max()
        # det_map = policy.logits
        # det_map_norm = th.sigmoid(det_map)

        if isinstance(policy_output, list):
            policy_output.append(policy)

        return det_map_norm

    def get_keypoints_from_detmap(
        self, det_map_norm: Tensor, max_kpts: float | int, thr: float = None
    ) -> MethodOutput:
        # ? extract keypoints as local maxima
        thr = thr if thr else self.thr_detector
        xy, scores = extract_maxima_from_map(
            det_map_norm[None], thr=thr, nms_radius=3, border=0, max_kpts=max_kpts
        )
        xy = xy[0]
        scores = scores[0]

        # ? add the border (the det_map is smaller than the original image)
        xy += self.model.border

        # ? put fake values for the rest
        kpts_sizes = self.model.receptive_field * th.ones_like(xy[:, 0])

        output = MethodOutput(
            kpts=xy,
            kpts_scores=scores,
            kpts_sizes=kpts_sizes,
            det_map=det_map_norm,
        )
        return output

    def load_image(self, path: str | Path, scaling: float = 1.0) -> Tensor:
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
        self, img: Tensor | np.ndarray, multiple_of: int = 16
    ) -> Tensor | np.ndarray:
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

    def move_to(self, device: str = "cpu") -> "MethodWrapper":
        """Move the model to the specified device."""
        self.device = device
        self.model.to(device)
        if self.descriptor_network is not None:
            self.descriptor_network.to(device)
        return self
