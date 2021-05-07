"""
Implementation of ResNet network family with
intermediate layers returned in forward step.
"""

# ==================== [IMPORT] ====================

import os
from typing import Callable, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# ==================== [CONFIG] ====================

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# ===================== [CODE] =====================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int,
                       out_channels: int,
                       stride: int = 1,
                       downsample: Callable = None):

        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int,
                       out_channels: int,
                       stride: int = 1,
                       downsample: Callable = None):

        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out



class ResNet(nn.Module):

    def __init__(self, block: Union[BasicBlock, Bottleneck],
                       layers: List[int]):

        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, 1000)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        x = self.avgpool(layer4_out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ===================== [UTILS] =====================


def load_weights(model_name: str):
    filename = f'weights/{model_name}.pth'

    if os.path.exists(filename):
        state = torch.load(filename)
        return state

    checkpoint_url = model_urls[model_name]
    return model_zoo.load_url(checkpoint_url)


def resnet18_pretrained():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(load_weights('resnet18'))
    return model


def resnet34_pretrained():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.load_state_dict(load_weights('resnet34'))
    return model


def resnet50_pretrained():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(load_weights('resnet50'))
    return model


def resnet101_pretrained():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.load_state_dict(load_weights('resnet101'))
    return model


def resnet152_pretrained():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    model.load_state_dict(load_weights('resnet152'))
    return model

