"""
Implementation of AInnoFace model (https://arxiv.org/pdf/1905.01585v3.pdf).
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np
from PIL import Image

from typing import (
    Union,
    List,
)

from .resnet import (
    resnet18_pretrained,
    resnet34_pretrained,
    resnet50_pretrained,
    resnet101_pretrained,
    resnet152_pretrained,
)

from .widerface import WIDERFACEImage
from .rfe import ReceptiveFieldEnrichment
from .anchor import generate_anchor_boxes

# ===================== [CODE] =====================


class AInnoFace(nn.Module):

    def __init__(self, backbone: str = 'resnet18',
                       num_anchors: int = 2,
                       interpolation_mode: str = 'nearest',
                       compute_first_step: bool = True):
        """
        Parameters
        ----------
        backbone
            Backbone network. One of 'resnet18', 'resnet32',
            'resnet50', 'resnet101', 'resnet152'.
        num_anchors
            Number of anchor boxes shift predictions to return.
            (Each prediction is tuple of 4 real numbers).
        interpolation_mode
            Mode to use in feature pyramid network to upsample
            tensors from upper levels.
        compute_first_step
            Compute first step proposals or not (computing them
            is not required in test mode).
        """

        super(AInnoFace, self).__init__()
        self.interpolation_mode = interpolation_mode
        self._compute_fs = compute_first_step

        # configuring backbone
        if backbone == 'resnet152':
            self._channels = 256
            self._backbone_channels = [256, 512, 1024, 2048]
            self.backbone = resnet152_pretrained()
        elif backbone == 'resnet101':
            self._channels = 256
            self._backbone_channels = [256, 512, 1024, 2048]
            self.backbone = resnet101_pretrained()
        elif backbone == 'resnet50':
            self._channels = 256
            self._backbone_channels = [256, 512, 1024, 2048]
            self.backbone = resnet50_pretrained()
        elif backbone == 'resnet34':
            self._channels = 128
            self._backbone_channels = [64, 128, 256, 512]
            self.backbone = resnet34_pretrained()
        elif backbone == 'resnet18':
            self._channels = 128
            self._backbone_channels = [64, 128, 256, 512]
            self.backbone = resnet18_pretrained()

        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # other bottom-up path layers
        self.raw_level5 = nn.Conv2d(self._backbone_channels[3], self._backbone_channels[3] // 2,
                                        kernel_size=3, stride=2, padding=1)
        self.raw_level6 = nn.Conv2d(self._backbone_channels[3] // 2, self._channels,
                                        kernel_size=3, stride=2, padding=1)

        # lateral connections of feature pyramid network
        self.fpn_lateral1 = nn.Conv2d(self._backbone_channels[0],  self._channels, kernel_size=1)
        self.fpn_lateral2 = nn.Conv2d(self._backbone_channels[1],  self._channels, kernel_size=1)
        self.fpn_lateral3 = nn.Conv2d(self._backbone_channels[2], self._channels, kernel_size=1)
        self.fpn_lateral4 = nn.Conv2d(self._backbone_channels[3], self._channels, kernel_size=1)

        # first stage of selective refinement network
        self.srn_fs_conv1 = nn.Conv2d(self._backbone_channels[0], self._channels, kernel_size=1)
        self.srn_fs_conv2 = nn.Conv2d(self._backbone_channels[1], self._channels, kernel_size=1)
        self.srn_fs_conv3 = nn.Conv2d(self._backbone_channels[2], self._channels, kernel_size=1)
        self.srn_fs_conv4 = nn.Conv2d(self._backbone_channels[3], self._channels, kernel_size=1)
        self.srn_fs_conv5 = nn.Conv2d(self._backbone_channels[3] // 2, self._channels, kernel_size=1)
        self.srn_fs_conv6 = nn.Conv2d(self._channels,  self._channels, kernel_size=1)

        # second stage of selective refinement network
        self.srn_ss_conv1 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1)
        self.srn_ss_conv2 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1)
        self.srn_ss_conv3 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1)
        self.srn_ss_conv4 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1)
        self.srn_ss_conv5 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1, stride=2)
        self.srn_ss_conv6 = nn.Conv2d(self._channels, self._channels, kernel_size=3, padding=1, stride=2)

        # classification head
        self.cls_head = nn.Sequential(
                ReceptiveFieldEnrichment(in_channels=self._channels),
                nn.Conv2d(self._channels, num_anchors, kernel_size=1),
            )

        # bbox regression head
        self.box_head = nn.Sequential(
                ReceptiveFieldEnrichment(in_channels=self._channels),
                nn.Conv2d(self._channels, num_anchors * 4, kernel_size=1),
            )

        # normalization transforms
        self.normalize = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), # also normalizes from 0-255 to 0-1
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    def _fpn_upsample(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, scale_factor=2, mode=self.interpolation_mode)


    def _preprocess_images(self, images: Union[torch.Tensor, np.ndarray,
                              List[Image.Image], List[WIDERFACEImage]]) -> torch.Tensor:
        """
        Convert input image format to torch tensor.
        """
        processed_images = []

        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image, dtype=np.uint8)

            elif isinstance(image, WIDERFACEImage):
                image = image.pixels(format='numpy')

            elif isinstance(image, np.ndarray):
                pass

            elif isinstance(image, torch.Tensor):
                image = image.numpy().astype(np.uint8)

            else:
                raise ValueError('Wrong type of input image')

            # normalize from 0-255 to 0-1 and convert to torch.Tensor
            processed_images.append(self.normalize(image))

        images = torch.stack(processed_images)
        return images


    @staticmethod
    def _flatten_pred_box(pred: torch.Tensor) -> torch.Tensor:
        """
        Flatten predictions of bbox head.
        """
        batch_size, _, _, _ = pred.shape
        return pred.reshape(batch_size, -1, 4)


    @staticmethod
    def _flatten_pred_cls(pred: torch.Tensor) -> torch.Tensor:
        """
        Flatten predictions of cls head.
        """
        batch_size, _, _, _ = pred.shape
        return pred.reshape(batch_size, -1, 1)


    @staticmethod
    def _flatten_anchors(anchors: torch.Tensor) -> torch.Tensor:
        """
        Flatten anchor boxes coordinates.
        """
        return anchors.view(-1, 4)


    def _move_anchors(self, anchors: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        """
        Move anchor boxes the way model predicts.
        """
        result = torch.zeros_like(proposals)
        result[..., 0] = proposals[..., 0] * anchors[:, 2] + anchors[:, 0] # prop_y * anch_h + anch_y
        result[..., 1] = proposals[..., 1] * anchors[:, 3] + anchors[:, 1] # prop_x * anch_w + anch_x
        result[..., 2] = torch.exp(proposals[..., 2]) * anchors[:, 2]      # exp(prop_h) * anch_h
        result[..., 3] = torch.exp(proposals[..., 3]) * anchors[:, 3]      # exp(prop_w) * anch_w
        return result


    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert (yc, xc, h, w) format to (y_up_left, x_up_left, w, h) format.
        """
        result = boxes.clone()
        result[..., 0] -= boxes[..., 2] / 2
        result[..., 1] -= boxes[..., 3] / 2
        return result


    def forward(self, images: Union[torch.Tensor, np.ndarray,
                              List[Image.Image], List[WIDERFACEImage]],
                      device: torch.device = 'cpu') -> torch.Tensor:
        """
        Make forward pass of AInnoFace model.

        Parameters
        ----------
        images
            Input images. Can be a batch of torch tensors of shape (B, H, W, 3),
            batch of numpy tensors of shape (B, H, W, 3), list of PIL images or list
            of WIDERFACEImage's. Each image is supposed to be a tensor of shape (H, W, 3),
            each value is uint8 in range 0-255.

        Returns
        -------
        Tuple of three values (fs_proposals: optional, ss_proposals, anchors)

        fs_proposals and ss_proposals:
            torch.Tensor of shape (batch_size, *, 6), each line contains
            description of bounding box in format: (y, x, h, w, prob, level).

        anchors:
            torch.Tensor of shape (*, 4), each line contains description
            of bounding box in format (y, x, h, w).
        """

        # converting any possible type to torch tensor and normalizing
        images = self._preprocess_images(images)
        images = images.to(device)
        batch_size, _, im_height, im_width = images.shape

        # extracting raw features
        raw_level1, raw_level2, raw_level3, raw_level4 = self.backbone(images)

        if self._compute_fs:
            raw_level5 = F.relu(self.raw_level5(raw_level4))
            raw_level6 = F.relu(self.raw_level6(raw_level5))

        # computing fpn features
        fpn_level4 = F.relu(self.fpn_lateral4(raw_level4))
        fpn_level3 = F.relu(self.fpn_lateral3(raw_level3)) + self._fpn_upsample(fpn_level4)
        fpn_level2 = F.relu(self.fpn_lateral2(raw_level2)) + self._fpn_upsample(fpn_level3)
        fpn_level1 = F.relu(self.fpn_lateral1(raw_level1)) + self._fpn_upsample(fpn_level2)

        # first stage of selective refinement network
        if self._compute_fs:
            srn_fs_level1 = F.relu(self.srn_fs_conv1(raw_level1))
            srn_fs_level2 = F.relu(self.srn_fs_conv2(raw_level2))
            srn_fs_level3 = F.relu(self.srn_fs_conv3(raw_level3))
            srn_fs_level4 = F.relu(self.srn_fs_conv4(raw_level4))
            srn_fs_level5 = F.relu(self.srn_fs_conv5(raw_level5))
            srn_fs_level6 = F.relu(self.srn_fs_conv6(raw_level6))

        # second stage of selective refinement network
        srn_ss_level1 = F.relu(self.srn_ss_conv1(fpn_level1))
        srn_ss_level2 = F.relu(self.srn_ss_conv2(fpn_level2))
        srn_ss_level3 = F.relu(self.srn_ss_conv3(fpn_level3))
        srn_ss_level4 = F.relu(self.srn_ss_conv4(fpn_level4))
        srn_ss_level5 = F.relu(self.srn_ss_conv5(fpn_level4))    # explained in SRN article
        srn_ss_level6 = F.relu(self.srn_ss_conv6(srn_ss_level5)) # explained in SRN article

        # computing head outputs
        if self._compute_fs:
            srn_fs = [srn_fs_level1, srn_fs_level2, srn_fs_level3, srn_fs_level4, srn_fs_level5, srn_fs_level6]
            cls_head_fs = [self.cls_head(x) for x in srn_fs]
            box_head_fs = [self.box_head(x) for x in srn_fs]

        srn_ss = [srn_ss_level1, srn_ss_level2, srn_ss_level3, srn_ss_level4, srn_ss_level5, srn_ss_level6]
        cls_head_ss = [self.cls_head(x) for x in srn_ss]
        box_head_ss = [self.box_head(x) for x in srn_ss]

        # computing anchor list and proposals for
        # first and second stages
        anchors = []
        proposals_fs = []
        proposals_ss = []

        for level in range(6):
            downsampling_factor = 2 ** (level + 2)
            level_height = im_height // downsampling_factor
            level_width = im_width // downsampling_factor

            level_anchors = generate_anchor_boxes(height=level_height,
                                                  width=level_width,
                                                  downsampling_factor=downsampling_factor,
                                                  aspect_ratios=[1.25],
                                                  scales=[2, 2 * np.sqrt(2)],
                                                  base_size=2)
            level_anchors = level_anchors.to(device)

            # shape anchors correctly
            anchors_original_shape = level_anchors.shape
            level_anchors = self._flatten_anchors(level_anchors)

            # second stage proposals
            _, ss_channels, ss_height, ss_width = box_head_ss[level].shape
            assert (ss_height, ss_width, ss_channels // 4, 4) == anchors_original_shape

            ss_level_cls = self._flatten_pred_cls(cls_head_ss[level].permute(0, 2, 3, 1))
            ss_level_box = self._flatten_pred_box(box_head_ss[level].permute(0, 2, 3, 1))
            ss_level_box = self._move_anchors(level_anchors, ss_level_box)
            ss_level_box = self._normalize_boxes(ss_level_box)
            ss_level_id = torch.full_like(ss_level_cls, fill_value=level + 1)
            ss_level_pred = torch.cat([ss_level_box, ss_level_cls, ss_level_id], dim=2)
            proposals_ss.append(ss_level_pred)

            # first stage proposals
            if self._compute_fs:
                _, fs_channels, fs_height, fs_width = box_head_fs[level].shape
                assert (fs_height, fs_width, fs_channels // 4, 4) == anchors_original_shape

                fs_level_cls = self._flatten_pred_cls(cls_head_fs[level].permute(0, 2, 3, 1))
                fs_level_box = self._flatten_pred_box(box_head_fs[level].permute(0, 2, 3, 1))
                fs_level_box = self._move_anchors(level_anchors, fs_level_box)
                fs_level_box = self._normalize_boxes(fs_level_box)
                fs_level_id = torch.full_like(fs_level_cls, fill_value=level + 1)
                fs_level_pred = torch.cat([fs_level_box, fs_level_cls, fs_level_id], dim=2)
                proposals_fs.append(fs_level_pred)

            # normalize anchors before append
            level_anchors = self._normalize_boxes(level_anchors)
            anchors.append(level_anchors)

        # concatinating tensors
        anchors = torch.cat(anchors, dim=0)
        proposals_ss = torch.cat(proposals_ss, dim=1)

        if self._compute_fs:
            proposals_fs = torch.cat(proposals_fs, dim=1)

        # return result
        if self._compute_fs:
            return proposals_fs, proposals_ss, anchors

        return proposals_ss, anchors

