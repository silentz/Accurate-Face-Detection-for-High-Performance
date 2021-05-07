"""
Receptive field enrichment layer implementation
from https://arxiv.org/pdf/1809.02693.pdf.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== [CODE] =====================


class ReceptiveFieldEnrichment(nn.Module):

    def __init__(self, in_channels: int,
                       inter_channels: int = 64):
        """
        Parameters
        ----------
        in_channels
            Count of channels in input tensor.
        inter_channels
            Count of internal operations channels. 64 by default.
        """

        super(ReceptiveFieldEnrichment, self).__init__()
        self._channels = inter_channels

        self.branch_1 = nn.Sequential(
                nn.Conv2d(in_channels,    self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 3), padding=(0, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
            )

        self.branch_2 = nn.Sequential(
                nn.Conv2d(in_channels,    self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 5), padding=(0, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
            )

        self.branch_3 = nn.Sequential(
                nn.Conv2d(in_channels,    self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(3, 1), padding=(1, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
            )

        self.branch_4 = nn.Sequential(
                nn.Conv2d(in_channels,    self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(5, 1), padding=(2, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._channels, self._channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
            )

        self.final = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0)),
                nn.ReLU(inplace=True),
            )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x1 = self.branch_1(X)
        x2 = self.branch_2(X)
        x3 = self.branch_3(X)
        x4 = self.branch_4(X)

        total = torch.cat([x1, x2, x3, x4], dim=1)
        result = X + self.final(total)
        return result

