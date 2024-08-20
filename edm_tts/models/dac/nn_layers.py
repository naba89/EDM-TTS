"""Modified from: https://github.com/descriptinc/descript-audio-codec"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def WNConv1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    # Scripting this brings model speed up 1.4x
    @staticmethod
    @torch.jit.script
    def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        x = x.reshape(shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel_size: int = 7) -> None:
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
