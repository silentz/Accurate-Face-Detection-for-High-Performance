"""
AInnoFace model loss layer.
"""

# ==================== [IMPORT] ==================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== [CODE] ===================


class AInnoFaceLoss(nn.Module):

    def __init__(self):
        super(self, AInnoFaceLoss).__init__()


    def forward(self, fs_proposal: torch.Tensor,
                      ss_proposal: torch.Tensor,
                      anchors: torch.Tensor,
                      ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fs_proposal: (batch_size, n_anchors, 5)
            First stage proposals of AInnoFace model.
        ss_proposal: (batch_size, n_anchors, 5)
            Second stage proposals of AInnoFace model.
        anchors: (n_anchors, 4)
            Original anchor boxes.
        ground_truth: (batch_size, n_boxes, 4)
            Ground truth bounding boxes for images.
        """

