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

