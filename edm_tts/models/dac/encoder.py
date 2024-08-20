"""Modified from: https://github.com/descriptinc/descript-audio-codec"""

import math

import torch
import torch.nn as nn

from .nn_layers import ResidualUnit, Snake1d, WNConv1d


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
    ) -> None:
        super().__init__()
        # Create first convolution
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_model, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
