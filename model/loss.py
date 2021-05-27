"""
AInnoFace model loss layer.
"""

# ==================== [IMPORT] ==================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops

from typing import List

# ===================== [CODE] ===================


class AInnoFaceLoss(nn.Module):

    def __init__(self, two_step: bool = True):
        super(AInnoFaceLoss, self).__init__()
        self.fs_low_threshold = 0.3
        self.ss_low_threshold = 0.4
        self.ss_high_threshold = 0.5
        self.fs_high_threshold = 0.7
        self.two_step = two_step


    @staticmethod
    def get_target_boxes_with_scores(anchors: torch.Tensor,
                                     ground_truth: List[torch.Tensor]) -> torch.Tensor:
        """
        Map anchor boxes and target boxes with each other and return
        IoU score for each found bbox.

        Parameters
        ----------
        anchors
            Torch tensor of shape (C, 4) containing anchor boxes in (x, y, w, h)
            format.
        ground_truth
            List of torch tensors of shape (K, 4) containing ground truth boxes for each image of batch
            in (x, y, w, h) format.

        Returns
        -------
        Two torch.Tensors, one containing coordinates of most probable ground
        truth box for each anchor box in format (x, y, w, h) and
        other containig IoU score for each box.
        """

        batch_size = len(ground_truth)
        corner_anchors = torchvision.ops.box_convert(anchors, in_fmt='xywh', out_fmt='xyxy')
        target_boxes = []
        target_iou_scores = []

        for image_id in range(batch_size):
            if len(ground_truth[image_id]) <= 0:
                zeros = torch.zeros(corner_anchors.shape[0])
                zeros = zeros.to(corner_anchors.device)
                target_boxes.append(corner_anchors)
                target_iou_scores.append(zeros)
                continue

            image_gt_boxes = torchvision.ops.box_convert(ground_truth[image_id], in_fmt='xywh', out_fmt='xyxy')
            iou_scores = torchvision.ops.box_iou(corner_anchors, image_gt_boxes)
            boxes_with_max_iou = torch.argmax(iou_scores, dim=1)
            image_gt_boxes = torchvision.ops.box_convert(image_gt_boxes, in_fmt='xyxy', out_fmt='xywh')

            target_boxes.append(image_gt_boxes[boxes_with_max_iou])
            target_iou_scores.append(torch.max(iou_scores, dim=1).values)

        target_boxes = torch.stack(target_boxes, dim=0)
        target_iou_scores = torch.stack(target_iou_scores, dim=0)

        return target_boxes, target_iou_scores


    @staticmethod
    def iou_loss(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
        boxes_1 = torchvision.ops.box_convert(boxes_1, in_fmt='xywh', out_fmt='xyxy')
        boxes_2 = torchvision.ops.box_convert(boxes_2, in_fmt='xywh', out_fmt='xyxy')
        loss = 0

        for box1, box2 in zip(boxes_1, boxes_2):
            iou = torchvision.ops.boxes.box_iou(
                    torch.unsqueeze(box1, 0),
                    torch.unsqueeze(box2, 0))
            loss += - torch.log(iou[0][0] + 1e-2)

        return loss


    def forward(self, ss_proposal: torch.Tensor,
                      anchors: torch.Tensor,
                      ground_truth: List[torch.Tensor],
                      fs_proposal: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        fs_proposal: (batch_size, n_anchors, 6)
            First stage proposals of AInnoFace model.
        ss_proposal: (batch_size, n_anchors, 6)
            Second stage proposals of AInnoFace model.
        anchors: (n_anchors, 4)
            Original anchor boxes.
        ground_truth: (batch_size, n_boxes, 4)
            Ground truth bounding boxes for images.
        """

        # TODO: add fs stage

        batch_size = ss_proposal.shape[0]
        target_boxes, target_score = self.get_target_boxes_with_scores(anchors, ground_truth)

        fs_stc_loss = 0
        fs_str_loss = 0
        ss_str_loss = 0
        ss_stc_loss = 0

        for image_id in range(batch_size):
            image_target_boxes = target_boxes[image_id]
            image_target_score = target_score[image_id]

            # second stage
            pred_ss_boxes = ss_proposal[image_id, :, :4]
            pred_ss_cls = ss_proposal[image_id, :, 4]

            ss_positive_mask = (image_target_score >= self.ss_high_threshold)
            ss_negative_mask = (image_target_score < self.ss_low_threshold)

            ss_positive_count = ss_positive_mask.sum().item()
            ss_negative_count = ss_negative_mask.sum().item()

            ss_local_stc_loss = 0
            ss_local_str_loss = 0

            if ss_positive_count > 0:
                local_target = torch.ones(ss_positive_count)
                local_target = local_target.to(ss_proposal.device)
                ss_local_stc_loss += torchvision.ops.sigmoid_focal_loss(
                                    inputs=pred_ss_cls[ss_positive_mask],
                                    targets=local_target,
                                    alpha=0.25,
                                    gamma=2,
                                    reduction='sum',
                                )

            if ss_negative_count > 0:
                local_target = torch.zeros(ss_negative_count)
                local_target = local_target.to(ss_proposal.device)
                ss_local_stc_loss += torchvision.ops.sigmoid_focal_loss(
                                    inputs=pred_ss_cls[ss_negative_mask],
                                    targets=local_target,
                                    alpha=0.25,
                                    gamma=2,
                                    reduction='sum',
                                )

            if ss_positive_count > 0:
                ss_local_stc_loss /= ss_positive_count

            if ss_positive_count > 0:
                iou_loss = self.iou_loss(pred_ss_boxes[ss_positive_mask], image_target_boxes[ss_positive_mask])
                iou_loss /= ss_positive_count
                ss_local_str_loss += iou_loss

            ss_stc_loss += ss_local_stc_loss
            ss_str_loss += ss_local_str_loss

        fs_stc_loss /= batch_size
        fs_str_loss /= batch_size
        ss_stc_loss /= batch_size
        ss_str_loss /= batch_size
        total = fs_stc_loss + fs_str_loss + ss_stc_loss + ss_str_loss
        return total

