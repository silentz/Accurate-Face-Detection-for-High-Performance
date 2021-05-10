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

    def __init__(self, num_anchors: int = 2,
                       interpolation_mode: str = 'nearest',
                       compute_first_step: bool = True):
        """
        Parameters
        ----------
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
        self._channels = 256
        self._backbone_channels = [256, 512, 1024, 2048]

        # bottom-up path layers
        self.backbone = resnet152_pretrained()
        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

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


    def _move_anchors(self, proposals: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Move anchor boxes the way model predicts.
        """
        result = torch.zeros_like(proposals)
        result[:, :, 4] = proposals[:, :, 4] # saving probabilities as they are
        result[:, :, 0] = proposals[:, :, 0] * anchors[:, 2] + anchors[:, 0]
        result[:, :, 1] = proposals[:, :, 1] * anchors[:, 3] + anchors[:, 1]
        result[:, :, 2] = torch.exp(proposals[:, :, 2]) * anchors[:, 2]
        result[:, :, 3] = torch.exp(proposals[:, :, 3]) * anchors[:, 3]
        return result


    def _normalize_boxes(self, proposals: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert boxes values to (x_up_left, y_up_left, w, h) and clip
        their values to image height and width.
        """
        # TODO
        return proposals


    def forward(self, images: Union[torch.Tensor, np.ndarray,
                              List[Image.Image], List[WIDERFACEImage]]) -> torch.Tensor:
        """
        Make forward pass of AInnoFace model.

        Parameters
        ----------
        images
            Input images. Can be a batch of torch tensors of shape (B, H, W, 3),
            batch of numpy tensors of shape (B, H, W, 3), list of PIL images or list
            of WIDERFACEImage's. Each image is supposed to be a tensor of shape (H, W, 3),
            each value is uint8 in range 0-255.
        """

        # converting any possible type to torch tensor and normalizing
        images = self._preprocess_images(images)
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
            cls_head_fs = [self.cls_head(x).permute(0, 2, 3, 1) for x in srn_fs]
            box_head_fs = [self.box_head(x).permute(0, 2, 3, 1) for x in srn_fs]

        srn_ss = [srn_ss_level1, srn_ss_level2, srn_ss_level3, srn_ss_level4, srn_ss_level5, srn_ss_level6]
        cls_head_ss = [self.cls_head(x).permute(0, 2, 3, 1) for x in srn_ss]
        box_head_ss = [self.box_head(x).permute(0, 2, 3, 1) for x in srn_ss]

        # computing anchor list for all levels
        anchor_all = []

        for level in range(6):
            downsampling_factor = 2 ** (level + 1)
            _, height, width, _ = box_head_ss[level].shape

            anchors = generate_anchor_boxes(height=height,
                                            width=width,
                                            downsampling_factor=downsampling_factor,
                                            aspect_ratios=[1.25],
                                            scales=[2, 2 * np.sqrt(2)],
                                            base_size=2)

            anchors = anchors.view(-1, 4)
            anchor_all.append(anchors)

        # computing proposals: each proposal has format (p, x, y, w, h)
        # where p is probability to be foreground
        proposals_fs = []
        proposals_ss = []

        for level in range(6):
            level_box_ss = box_head_ss[level].reshape(batch_size, -1, 4)
            level_cls_ss = F.sigmoid(cls_head_ss[level].reshape(batch_size, -1, 1))
            level_pred_ss = torch.cat([level_box_ss, level_cls_ss], dim=2)
            proposals_ss.append(level_pred_ss)

            if self._compute_fs:
                level_box_fs = box_head_fs[level].reshape(batch_size, -1, 4)
                level_cls_fs = F.sigmoid(cls_head_fs[level].reshape(batch_size, -1, 1))
                level_pred_fs = torch.cat([level_box_fs, level_cls_fs], dim=2)
                proposals_fs.append(level_pred_fs)

        anchor_all = torch.cat(anchor_all, dim=0)
        proposals_ss = torch.cat(proposals_ss, dim=1)
        if self._compute_fs:
            proposals_fs = torch.cat(proposals_fs, dim=1)

        # patching original anchor boxes with predictions of model
        proposals_ss = self._move_anchors(proposals_ss, anchor_all)
        proposals_ss = self._normalize_boxes(proposals_ss)
        if self._compute_fs:
            proposals_fs = self._move_anchors(proposals_fs, anchor_all)
            proposals_fs = self._normalize_boxes(proposals_fs)

        if self._compute_fs:
            return proposals_fs, proposals_ss, anchor_all

        return proposals_ss, anchor_all

