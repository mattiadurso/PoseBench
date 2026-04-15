import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import logging

try:
    from modules import Unet_down_block, Unet_up_block
except ImportError:
    from sandesc_models.sandesc.modules import Unet_down_block, Unet_up_block

# sys.path.append("/home/mattia/Desktop/Repos/dinov3")
# from dino_wrapper import DinoWrapper


class SANDesc(nn.Module):
    def __init__(
        self,
        ch_in: int = 3,
        kernel_size: int = 5,
        activ: str = "gelu",
        norm: str = "batch",
        skip_connection: bool = False,
        spatial_attention: bool = False,
        third_block: bool = False,
        down_output_channels=[16, 32, 64, 64, 64],
        up_output_channels=[64, 64, 64, 128],
        **kwargs,
    ):
        """
        Args:
            ch_in: int, number of input channels
            kernel_size: int, kernel size of the convolutional layers
            activ: str, activation function. Choose between 'relu', 'prelu', 'gelu'
            norm: str, normalization type
            skip_connection: bool, if True, skip connections and a second unet
                block are added to the network
            spatial_attention: bool, if True, spatial attention is added to the
                network
            third_block: bool, if True, adds a third block
            down_output_channels: list, number of channels of the output of
                each down block.
            up_output_channels: list, number of channels of the output of each
                up block. add +1 to get the same unet of disk in last element,
                eg [64, 64, 64, 128+1]

        Returns:
            des_vol: Tensor, descriptor volume. Shape [B, des_dim, H, W]
            x5: Tensor, output of the 5th down block. Shape [B, C5, H/8, W/8]

        The last element of up_output_channels is the number of channels of
        the descriptor.
        """

        super().__init__()
        self.conv_highest = nn.Conv2d(
            ch_in,
            down_output_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

        common = {
            "kernel_size": kernel_size,
            "activ": activ,
            "norm": norm,
            "skip_connection": skip_connection,  # and second block
            "spatial_attention": spatial_attention,
            "third_block": third_block,
        }

        self.down0 = Unet_down_block(
            down_output_channels[0], down_output_channels[1], **common
        )
        self.down1 = Unet_down_block(
            down_output_channels[1], down_output_channels[2], **common
        )
        self.down2 = Unet_down_block(
            down_output_channels[2], down_output_channels[3], **common
        )
        self.down3 = Unet_down_block(
            down_output_channels[3], down_output_channels[4], **common
        )

        self.up0 = Unet_up_block(
            down_output_channels[-1] + down_output_channels[-2],
            up_output_channels[0],
            **common,
        )
        self.up1 = Unet_up_block(
            down_output_channels[-3] + up_output_channels[0],
            up_output_channels[1],
            **common,
        )
        self.up2 = Unet_up_block(
            down_output_channels[-4] + up_output_channels[1],
            up_output_channels[2],
            **common,
        )
        self.up3 = Unet_up_block(
            down_output_channels[-5] + up_output_channels[2],
            up_output_channels[3],
            kernel_size=kernel_size,
            activ=None,
            norm=None,
        )

    def load_weights(self, weights):
        """
        Load weights into the model.
        Args:
            weights (str): Path to the weights file.
        """
        weights = torch.load(weights, weights_only=False)
        self.load_state_dict((weights["state_dict"]))
        logging.info(f"SANDesc weights loaded from {weights}.")

    def forward(self, x: Tensor, _=None) -> Tuple[Tensor, Tensor]:
        x0 = self.conv_highest(x)  # B,c_in,H,W

        x1 = self.down0(x0)  # B,C1,H/2,W/2
        x2 = self.down1(x1)  # B,C2,H/4,W/4
        x3 = self.down2(x2)  # B,C3,H/8,W/8
        x4 = self.down3(x3)  # B,C4,H/16,W/16

        x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
        x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
        x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
        x8 = self.up3(x7, x0)  # B,des_dim,H,W

        return F.normalize(x8, p=2, dim=1)  # B,des_dim,H,W


# class SANDescD(nn.Module):
#     def __init__(
#         self,
#         ch_in: int = 3,
#         kernel_size: int = 5,
#         activ: str = "gelu",
#         norm: str = "batch",
#         skip_connection: bool = False,
#         spatial_attention: bool = False,
#         third_block: bool = False,
#         down_output_channels=[16, 32, 64, 64, 64],
#         up_output_channels=[64, 64, 64, 128],
#         **kwargs
#     ):
#         """
#         Args:
#             ch_in: int, number of input channels
#             kernel_size: int, kernel size of the convolutional layers
#             activ: str, activation function. Choose between 'relu', 'prelu', 'gelu'
#             skip_connection: bool, if True, skip connections and a second unet block are added to the network
#             spatial_attention: bool, if True, spatial attention is added to the network
#             down_output_channels: list, number of channels of the output of each down block.
#             up_output_channels: list, number of channels of the output of each up block. add +1 to get the same unet of disk in last element, eg [64, 64, 64, 128+1]
#         Returns:
#             des_vol: Tensor, descriptor volume. Shape [B, des_dim, H, W]
#             x5: Tensor, output of the 5th down block. Shape [B, C5, H/8, W/8]

#         The last element of up_output_channels is the number of channels of the descriptor.
#         """

#         super().__init__()
#         self.conv_highest = nn.Conv2d(
#             ch_in,
#             down_output_channels[0],
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#             stride=1,
#         )

#         common = {
#             "kernel_size": kernel_size,
#             "activ": activ,
#             "norm": norm,
#             "skip_connection": skip_connection,  # and second block
#             "spatial_attention": spatial_attention,
#             "third_block": third_block,
#         }

#         self.down0 = Unet_down_block(
#             down_output_channels[0], down_output_channels[1], **common
#         )
#         self.down1 = Unet_down_block(
#             down_output_channels[1], down_output_channels[2], **common
#         )
#         self.down2 = Unet_down_block(
#             down_output_channels[2], down_output_channels[3], **common
#         )
#         # self.down3 = Unet_down_block(down_output_channels[3], down_output_channels[4], **common)

#         self.up0 = Unet_up_block(
#             down_output_channels[-1] + down_output_channels[-2],
#             up_output_channels[0],
#             **common
#         )
#         self.up1 = Unet_up_block(
#             down_output_channels[-3] + up_output_channels[0],
#             up_output_channels[1],
#             **common
#         )
#         self.up2 = Unet_up_block(
#             down_output_channels[-4] + up_output_channels[1],
#             up_output_channels[2],
#             **common
#         )
#         self.up3 = Unet_up_block(
#             down_output_channels[-5] + up_output_channels[2],
#             up_output_channels[3],
#             kernel_size=kernel_size,
#             activ=None,
#             norm=None,
#         )

#         self.dino = DinoWrapper(
#             model="s",
#             layer=5,
#             device="cuda" if torch.cuda.is_available() else "cpu",
#             feat_matching=False,
#         )
#         C = self.dino(torch.randn(1, 3, 224, 224).to(self.dino.device)).shape[1]
#         self.adapth_dino = nn.Conv2d(
#             C, down_output_channels[-1], kernel_size=1
#         )  # correct name here and in saved weights

#     def load_weights(self, weights):
#         """
#         Load weights into the model.
#         Args:
#             weights (str): Path to the weights file.
#         """
#         weights = torch.load(weights, weights_only=False)
#         self.load_state_dict((weights["state_dict"]))

#     def forward(self, x: Tensor, _=None) -> Tuple[Tensor, Tensor]:

#         x4 = self.adapth_dino(self.dino(x))  # B,C4,H/16,W/16
#         x0 = self.conv_highest(x)  # B,c_in,H,W

#         x1 = self.down0(x0)  # B,C1,H/2,W/2
#         x2 = self.down1(x1)  # B,C2,H/4,W/4
#         x3 = self.down2(x2)  # B,C3,H/8,W/8
#         # x4 = self.down3(x3)  # B,C4,H/16,W/16

#         x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
#         x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
#         x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
#         x8 = self.up3(x7, x0)  # B,des_dim,H,W

#         x8 = F.normalize(x8, p=2, dim=1)  # B,des_dim,H,W

#         return x8
