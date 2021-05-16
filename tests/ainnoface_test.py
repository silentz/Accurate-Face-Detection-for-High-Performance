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

