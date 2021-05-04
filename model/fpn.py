"""
Feature pyramid network layer implementation.
"""

# ==================== [IMPORT] ====================

import copy
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== [CODE] =====================


class FeaturePyramid(nn.Module):
    """
    Feature pyramid network layer implementation.
    """

    def __init__(self, n_levels: int,
                       in_channels: Union[int, List[int]],
                       out_channels: int = 256,
                       interpolate_scale_factor: float = 2,
                       interpolate_mode: str = 'nearest'):
        """
        Parameters
        ----------
        n_levels
            Count of feature pyramid network layers.
        in_channels
            List of input channel count for each FPN layer.
            Can be constant value, if input for each layer
            has the same channel count.
        out_channels
            Channel count of feature pyramid network
            result in each layer.
        interpolate_scale_factor
            Scale factor of each interpolation procedure
            between layers.
        interpolate_mode
            Interpolation mode.
        """

        self.n_levels = n_levels
        self.in_channels = copy.deepcopy(in_channels)
        self.out_channels = copy.deepcopy(out_channels)
        self.scale_factor = interpolate_scale_factor
        self.interpolate_mode = interpolate_mode

        if not isinstance(in_channels, list):
            self.in_channels = [in_channels for _ in range(n_levels)]

        self._input_layers = nn.ModuleList(
                nn.ModuleList(
                    nn.Conv2d(
                        in_channels=in_channels[idx],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False, #  batchnorm is used instead
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                for idx in range(n_levels)
            )

        self._merge_layers = nn.ModuleList(
                nn.ModuleList(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False, # batchnorm is used instead
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                for _ in range(n_levels - 1)
            )


    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:
        """
        Run input tensors through feature pyramid network.

        Parameters
        ----------
        input
            List of input tensors. Length of list should be equal
            to `n_levels`.
        """

        intermediate = [f(x) for f, x in zip(self._input_layers, input)]
        intermediate = intermediate[::-1]

        for idx in range(self.n_levels - 1):
            upsampled = F.interpolate(intermediate[idx], scale_factor=self.scale_factor,
                                        mode=self.interpolate_mode)
            raw_merged = upsampled + intermediate[idx + 1]
            intermediate[idx + 1] = self._merge_layers[idx](raw_merged)

        return intermediate[::-1]

