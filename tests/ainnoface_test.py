"""
Test suite for `model/ainnoface.py` module.
"""

# ==================== [IMPORT] ====================

import torch
import pytest
import numpy as np

from ..model.ainnoface import (
    AInnoFace
)

# ===================== [TEST] =====================


class TestAInnoFace:

    def test_normalization(self):
        torch.manual_seed(42)

        batch_size, height, width = 2, 3, 4
        box_channels, cls_channels = 8, 2

        unnormalized_box = torch.randn(batch_size, height, width, box_channels)
        unnormalized_cls = torch.randn(batch_size, height, width, cls_channels)
        unnormalized_anchors = torch.randn(height, width, cls_channels, 4)

        # preparing computationally slow but sure result (box and cls)
        prep_norm_box = []
        prep_norm_cls = []
        prep_norm_anchors = []

        for b in range(batch_size):
            batch_box = []
            batch_cls = []

            for h in range(height):
                for w in range(width):
                    for t in range(cls_channels):
                        box = unnormalized_box[b][h][w][t * 4 : (t + 1) * 4]
                        cls = torch.Tensor([unnormalized_cls[b][h][w][t]])
                        batch_box.append(box)
                        batch_cls.append(cls)

            batch_box = torch.stack(batch_box, dim=0)
            batch_cls = torch.stack(batch_cls, dim=0)
            prep_norm_box.append(batch_box)
            prep_norm_cls.append(batch_cls)

        prep_norm_box = torch.stack(prep_norm_box, dim=0)
        prep_norm_cls = torch.stack(prep_norm_cls, dim=0)

        # preparing computationally slow but sure anchors
        for h in range(height):
            for w in range(width):
                for t in range(cls_channels):
                    anch = unnormalized_anchors[h][w][t]
                    prep_norm_anchors.append(anch)

        prep_norm_anchors = torch.stack(prep_norm_anchors, dim=0)

        # comparing to normalization functions
        assert torch.allclose(prep_norm_box, AInnoFace._flatten_pred_box(unnormalized_box))
        assert torch.allclose(prep_norm_cls, AInnoFace._flatten_pred_cls(unnormalized_cls))
        assert torch.allclose(prep_norm_anchors, AInnoFace._flatten_anchors(unnormalized_anchors))


    def test_normalization_2(self):
        pred_box = torch.Tensor([[
                [
                    [0., 0., 0., 0., 1., 1., 1., 1.],
                    [2., 2., 2., 2., 3., 3., 3., 3.],
                ],
                [
                    [4., 4., 4., 4., 5., 5., 5., 5.],
                    [6., 6., 6., 6., 7., 7., 7., 7.],
                ]
            ]])

        pred_cls = torch.Tensor([[
                [
                    [0., 1.],
                    [2., 3.],
                ],
                [
                    [4., 5.],
                    [6., 7.],
                ],
            ]])

        init_anchors = torch.Tensor([
                [
                    [[0., 0., 0., 0.], [1., 1., 1., 1.]],
                    [[2., 2., 2., 2.], [3., 3., 3., 3.]],
                ],
                [
                    [[4., 4., 4., 4.], [5., 5., 5., 5.]],
                    [[6., 6., 6., 6.], [7., 7., 7., 7.]],
                ],
            ])

        post_box = AInnoFace._flatten_pred_box(pred_box)
        post_cls = AInnoFace._flatten_pred_cls(pred_cls)
        post_anc = AInnoFace._flatten_anchors(init_anchors)

        assert torch.allclose(post_box, torch.Tensor([[
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [4., 4., 4., 4.],
                [5., 5., 5., 5.],
                [6., 6., 6., 6.],
                [7., 7., 7., 7.],
            ]]))

        assert torch.allclose(post_cls, torch.Tensor([[
                [0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.],
            ]]))

        assert torch.allclose(post_anc, torch.Tensor([
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [4., 4., 4., 4.],
                [5., 5., 5., 5.],
                [6., 6., 6., 6.],
                [7., 7., 7., 7.],
            ]))

