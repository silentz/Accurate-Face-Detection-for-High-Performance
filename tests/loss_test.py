"""
Test suite for `model/ainnoface.py` module.
"""

# ==================== [IMPORT] ====================

import torch
import pytest

from ..model.loss import (
    AInnoFaceLoss
)

# ===================== [TEST] =====================


class TestLoss:

    def test_target_box_matching(self):
        anchors = torch.Tensor([
                [1., 1., 1., 1.],
                [2., 1., 2., 2.],
                [3., 1., 1., 1.],
                [2., 2., 1., 1.],
                [1., 1., 1.5, 1.5],
            ])

        ground_truth_boxes = [torch.Tensor([
                [1., 1., 1., 1.],
                [2., 2., 1., 1.],
            ])]

        target_box, target_score = AInnoFaceLoss.get_target_boxes_with_scores(anchors, ground_truth_boxes)

        assert torch.allclose(target_box, torch.Tensor([[
                [1., 1., 1., 1.],
                [2., 2., 1., 1.],
                [1., 1., 1., 1.],
                [2., 2., 1., 1.],
                [1., 1., 1., 1.],
            ]]))

        assert torch.allclose(target_score, torch.Tensor([[
                1.0000, 0.2500, 0.0000, 1.0000, 0.444444444
            ]]))

