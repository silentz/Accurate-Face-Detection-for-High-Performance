"""
AInnoFace model loss layer.
"""

# ==================== [IMPORT] ==================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops

# ===================== [CODE] ===================


class AInnoFaceLoss(nn.Module):

    def __init__(self):
        super(AInnoFaceLoss, self).__init__()
        self.fs_low_threshold = 0.3
        self.ss_low_threshold = 0.4
        self.ss_high_threshold = 0.5
        self.fs_high_threshold = 0.7


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

        batch_size = ground_truth.shape[0]
        all_target_bbox = []
        all_target_score = []

        for image_id in range(batch_size):
            image_anchors = torchvision.ops.box_convert(anchors[image_id], in_fmt='xywh', out_fmt='xyxy')
            image_ground_truth = torchvision.ops.box_convert(ground_truth[image_id], in_fmt='xywh', out_fmt='xyxy')

            iou_score = torchvision.ops.box_iou(image_anchors, image_ground_truth)
            max_score_box = torch.argmax(iou_score, dim=1)

            target_bbox = image_ground_truth[max_score_box]
            target_score = torch.max(iou_score, dim=1).values

            all_target_bbox.append(target_bbox)
            all_target_score.append(target_score)

        all_target_score = torch.stack(all_target_score, dim=0)
        all_target_bbox = torch.stack(all_target_bbox, dim=0)

        print(all_target_score)
        print(all_target_bbox)

        # TODO: add level for bboxes in ainnoface...

        #  fs_boxes = fs_proposal[:, :, 0:4]
        #  fs_scores = fs_proposal[:, :, 4]

        #  ss_boxes = ss_proposal[:, :, 0:4]
        #  ss_scores = ss_proposal[:, :, 4]

