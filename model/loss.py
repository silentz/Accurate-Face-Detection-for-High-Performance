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
            Torch tensor of shape (C, 4) containing anchor boxes in (left_up_y, left_up_x, w, h)
            format.
        ground_truth
            List of torch tensors of shape (K, 4) containing ground truth boxes for each image of batch
            in (left_up_y, left_up_x, w, h) format.

        Returns
        -------
        Two torch.Tensors, one containing coordinates of most probable ground
        truth box for each anchor box in format (left_up_y, left_up_x, w, h) and
        other containig IoU score for each box.
        """

        batch_size = len(ground_truth)
        corner_anchors = torchvision.ops.box_convert(anchors, in_fmt='xywh', out_fmt='xyxy')
        target_boxes = []
        target_iou_scores = []

        for image_id in range(batch_size):
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
        count = 0

        for box1, box2 in zip(boxes_1, boxes_2):
            iou = torchvision.ops.boxes.box_iou(
                    torch.unsqueeze(box1, 0),
                    torch.unsqueeze(box2, 0))
            loss += - torch.log(iou[0][0])
            count += 1

        if count == 0:
            count = 1

        return loss


    def forward(self, fs_proposal: torch.Tensor,
                      ss_proposal: torch.Tensor,
                      anchors: torch.Tensor,
                      ground_truth: List[torch.Tensor]) -> torch.Tensor:
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

        target_boxes, target_scores = self.get_target_boxes_with_scores(anchors, ground_truth)

        # flatten structure
        target_boxes = target_boxes.view(-1, 4)
        target_scores = target_scores.view(-1)
        fs_proposal = fs_proposal.reshape(-1, 6)
        ss_proposal = ss_proposal.reshape(-1, 6)

        # calulating stage ids
        ss_pos = (target_scores >= self.ss_high_threshold)
        ss_neg = (target_scores <= self.ss_low_threshold)

        fs_pos = (target_scores >= self.fs_high_threshold)
        fs_neg = (target_scores <= self.fs_low_threshold)

        ss_pos_count = ss_pos.sum().item()
        fs_pos_count = fs_pos.sum().item()

        if ss_pos_count < 1e-1: ss_pos_count = 1
        if fs_pos_count < 1e-1: fs_pos_count = 1

        # stc loss
        ss_stc_loss = torchvision.ops.sigmoid_focal_loss(
                        ss_proposal[:, 4], target_scores, reduction='mean') / ss_pos_count
        fs_stc_loss = torchvision.ops.sigmoid_focal_loss(
                        fs_proposal[:, 4], target_scores, reduction='mean') / fs_pos_count

        # str loss
        ss_str_loss = self.iou_loss(ss_proposal[ss_pos][:, 0:4], target_boxes[ss_pos]) / ss_pos_count
        fs_str_loss = self.iou_loss(fs_proposal[fs_pos][:, 0:4], target_boxes[fs_pos]) / fs_pos_count

        return ss_stc_loss + fs_stc_loss + ss_str_loss + fs_str_loss

