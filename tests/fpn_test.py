"""
Test suite for `model/fpn.py` module.
"""

# ==================== [IMPORT] ====================

import torch
import pytest

from ..model.fpn import (
    FeaturePyramidNetwork,
)

# ===================== [TEST] =====================


class TestFeaturePyramidNetwork:

    def test_init(self):
        test_in_channels = [1, 2, 3, 4, 5, 6]
        fpn = FeaturePyramidNetwork(
                n_levels=6,
                in_channels=test_in_channels,
                out_channels=16,
                scale_factor=2,
                interpolation_mode='nearest',
            )

        # check parameters
        assert fpn.n_levels == 6
        assert fpn.in_channels == test_in_channels
        assert fpn.out_channels == 16
        assert fpn.scale_factor == 2
        assert fpn.interpolation_mode == 'nearest'

        # check input layers
        assert len(fpn._input_layers) == 6

        for idx, layer in enumerate(fpn._input_layers):
            assert layer[0].kernel_size == (1, 1)
            assert layer[0].in_channels == test_in_channels[idx]
            assert layer[0].out_channels == 16
            assert layer[1].num_features == 16

        # check merge layers
        assert len(fpn._merge_layers) == 5

        for idx, layer in enumerate(fpn._merge_layers):
            assert layer[0].kernel_size == (3, 3)
            assert layer[0].in_channels == 16
            assert layer[0].out_channels == 16
            assert layer[1].num_features == 16


    def test_forward(self):
        fpn = FeaturePyramidNetwork(
                n_levels=2,
                in_channels=1,
                out_channels=8,
            )

        input_tensors = [
            torch.Tensor([[[
                [2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2.],
            ]]]),
            torch.Tensor([[[
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
            ]]]),
        ]

        output = fpn(input_tensors)
        assert len(output) == 2
        assert output[0].shape == (1, 8, 6, 6)
        assert output[1].shape == (1, 8, 3, 3)

