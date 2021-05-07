"""
Implementation of AInnoFace model (https://arxiv.org/pdf/1905.01585v3.pdf).
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ===================== [CODE] =====================


class AInnoFace(nn.Module):

    def __init__(self, num_anchors: int = 2,
                       interpolation_mode: str = 'nearest'):
        """
        Parameters
        ----------
        num_anchors
            Number of anchor boxes shift predictions to return.
            (Each prediction is tuple of 4 real numbers).
        interpolation_mode
            Mode to use in feature pyramid network to upsample
            tensors from upper levels.
        """

        super(AInnoFace, self).__init__()
        self.interpolation_mode = interpolation_mode
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


    def _fpn_upsample(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, scale_factor=2, mode=self.interpolation_mode)


    def forward(self, images: Union[torch.Tensor, np.ndarray,
                              List[Image.Image], List[WIDERFACEImage]]) -> torch.Tensor:
        """
        Make forward pass of AInnoFace model.

        Parameters
        ----------
        images
            Input images. Can be a batch of torch tensors,
            batch of numpy tensors, list of PIL images or list of WIDERFACEImage's.
            Each image is supposed to be a tensor of shape (H, W, 3), each
            value in range 0-255.
        """

        # converting any possible type to torch tensor

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images, dtype=torch.float32)

        if isinstance(images, List):
            new_images = []

            for image in images:
                if isinstance(image, Image.Image):
                    new_images.append(torch.from_numpy(np.array(image, dtype=np.float32)))
                elif isinstance(image, WIDERFACEImage):
                    new_images.append(image.pixels(format='torch'))
                elif isinstance(image, np.ndarray):
                    new_images.append(torch.from_numpy(image, dtype=torch.float32))
                elif isinstance(image, torch.Tensor):
                    new_images.append(image)
                else:
                    raise ValueError('Wrong type of input image')

            images = torch.vstack(new_images)

        if not isinstance(images, torch.Tensor):
            raise ValueError('Invalid input format')

        # TODO: normalize images

        # extracting raw features
        raw_level1, raw_level2, raw_level3, raw_level4 = self.backbone(images)
        raw_level5 = F.relu(self.raw_level5(raw_level4))
        raw_level6 = F.relu(self.raw_level6(raw_level5))

        # computing fpn features
        fpn_level4 = F.relu(self.fpn_lateral4(raw_level4))
        fpn_level3 = F.relu(self.fpn_lateral3(raw_level3)) + self._fpn_upsample(fpn_level4)
        fpn_level2 = F.relu(self.fpn_lateral2(raw_level2)) + self._fpn_upsample(fpn_level3)
        fpn_level1 = F.relu(self.fpn_lateral1(raw_level1)) + self._fpn_upsample(fpn_level2)

        # first stage of selective refinement network
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
        srn_fs = [srn_fs_level1, srn_fs_level2, srn_fs_level3, srn_fs_level4, srn_fs_level5, srn_fs_level6]
        srn_ss = [srn_ss_level1, srn_ss_level2, srn_ss_level3, srn_ss_level4, srn_ss_level5, srn_ss_level6]

        cls_head_fs_out = [self.cls_head(x) for x in srn_fs]
        cls_head_ss_out = [self.cls_head(x) for x in srn_ss]

        box_head_fs_out = [self.box_head(x) for x in srn_fs]
        box_head_ss_out = [self.box_head(x) for x in srn_ss]

        return cls_head_fs_out, box_head_fs_out, cls_head_ss_out, box_head_ss_out


