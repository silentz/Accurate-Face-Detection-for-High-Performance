"""
Anchor boxes generating utilities.
"""

# ==================== [IMPORT] ====================

import torch
import numpy as np

from typing import (
    List,
)

# ===================== [CODE] =====================


def generate_anchor_boxes(height: int,
                          width: int,
                          downsampling_factor: float,
                          aspect_ratios: List[float] = [1.],
                          scales: List[float] = [1.],
                          base_size: float = 1) -> torch.Tensor:
    """
    Generate `height` x `width` grid of anchor boxes, with
    all possible combinations of `aspect_ratios` and `scales`.

    Parameters
    ----------
    height
        Height of grid.
    width
        Width of grid.
    downsampling_factor
        Downsampling factor of current layer. Is used to calculate
        anchor box scale.
    aspect_ratios
        List of all required aspect ratios for each anchor box of grid.
        (aspect ratio = height / width)
    scales
        List of all required scales for each anchor box of grid.
    base_size
        Anchor box size with downsampling factor and scale equal to 1.

    Returns
    -------
    torch.Tensor with following format: (H, W, C, 4), where C
    is amount of generated anchor boxes for each point, H is grid height,
    W is grid width and last 4 numbers are anchor box coordinates: (xc, yc, w, h).
        xc - x coordinate of anchor box center.
        yc - y coordinate of anchor box center.
        w - width of anchor box.
        h - height of anchor box.
    """

    if height <= 0 or width <= 0:
        return torch.Tensor()

    # center coordinates of each anchor box
    xc = np.arange(width) * downsampling_factor + downsampling_factor / 2
    yc = np.arange(height) * downsampling_factor + downsampling_factor / 2

    # aspect rations and scales
    ar = np.array(aspect_ratios)
    sc = np.array(scales)

    # all possible combinations
    xc, yc, ar, sc = np.meshgrid(xc, yc, ar, sc)
    xc = np.expand_dims(xc.ravel(), axis=1)
    yc = np.expand_dims(yc.ravel(), axis=1)
    ar = np.expand_dims(ar.ravel(), axis=1)
    sc = np.expand_dims(sc.ravel(), axis=1)

    # calculating width and height
    h = (base_size * downsampling_factor) * sc * np.sqrt(ar)
    w = (base_size * downsampling_factor) * sc / np.sqrt(ar)

    result = np.hstack([xc, yc, w, h])
    result = result.reshape(height, width, -1, 4)
    return torch.from_numpy(result).type(torch.float32)

