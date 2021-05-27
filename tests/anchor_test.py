"""
Test suite for `model/anchor.py` module.
"""

# ==================== [IMPORT] ====================

import torch
import pytest
import numpy as np

from ..model.anchor import (
    generate_anchor_boxes
)

# ===================== [TEST] =====================


class TestAnchorBoxes:

    def test_shapes(self):
        # equal size
        anchors = generate_anchor_boxes(height=10,
                                        width=10,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == (10, 10, 1, 4)

        # height > weidth
        anchors = generate_anchor_boxes(height=20,
                                        width=10,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == (20, 10, 1, 4)

        # width > height
        anchors = generate_anchor_boxes(height=10,
                                        width=20,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == (10, 20, 1, 4)

        # height == 0
        anchors = generate_anchor_boxes(height=0,
                                        width=20,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == torch.Size([0])

        # width == 0
        anchors == generate_anchor_boxes(height=0,
                                        width=20,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == torch.Size([0])

        # multiple aspect ratios and scales
        anchors = generate_anchor_boxes(height=10,
                                        width=20,
                                        downsampling_factor=2,
                                        aspect_ratios=[1., 2., 3.],
                                        scales=[1., 2., 3.],
                                        base_size=1.)

        assert isinstance(anchors, torch.Tensor)
        assert anchors.shape == (10, 20, 9, 4)


    def test_downsampling_factor(self):
        # df == 2
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=2,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[1., 1., 0.5, 0.5]],
                    [[3., 1., 0.5, 0.5]],
                    [[5., 1., 0.5, 0.5]],
                ],
                [
                    [[1., 3., 0.5, 0.5]],
                    [[3., 3., 0.5, 0.5]],
                    [[5., 3., 0.5, 0.5]],
                ],
            ]))

        # df == 1
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.]],
                    [[1.5, 0.5, 1., 1.]],
                    [[2.5, 0.5, 1., 1.]],
                ],
                [
                    [[0.5, 1.5, 1., 1.]],
                    [[1.5, 1.5, 1., 1.]],
                    [[2.5, 1.5, 1., 1.]],
                ],
            ]))


    def test_scales(self):
        # one scale
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.]],
                    [[1.5, 0.5, 1., 1.]],
                    [[2.5, 0.5, 1., 1.]],
                ],
                [
                    [[0.5, 1.5, 1., 1.]],
                    [[1.5, 1.5, 1., 1.]],
                    [[2.5, 1.5, 1., 1.]],
                ],
            ]))

        # two scales
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1., 2.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.], [0.5, 0.5, 2., 2.]],
                    [[1.5, 0.5, 1., 1.], [1.5, 0.5, 2., 2.]],
                    [[2.5, 0.5, 1., 1.], [2.5, 0.5, 2., 2.]],
                ],
                [
                    [[0.5, 1.5, 1., 1.], [0.5, 1.5, 2., 2.]],
                    [[1.5, 1.5, 1., 1.], [1.5, 1.5, 2., 2.]],
                    [[2.5, 1.5, 1., 1.], [2.5, 1.5, 2., 2.]],
                ],
            ]))


    def test_aspect_ratios(self):
        # one aspect ratio
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.]],
                    [[1.5, 0.5, 1., 1.]],
                    [[2.5, 0.5, 1., 1.]],
                ],
                [
                    [[0.5, 1.5, 1., 1.]],
                    [[1.5, 1.5, 1., 1.]],
                    [[2.5, 1.5, 1., 1.]],
                ],
            ]))

        # two aspect ratios
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1., 2.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.], [0.5, 0.5, 0.7071, 1.4142]],
                    [[1.5, 0.5, 1., 1.], [1.5, 0.5, 0.7071, 1.4142]],
                    [[2.5, 0.5, 1., 1.], [2.5, 0.5, 0.7071, 1.4142]],
                ],
                [
                    [[0.5, 1.5, 1., 1.], [0.5, 1.5, 0.7071, 1.4142]],
                    [[1.5, 1.5, 1., 1.], [1.5, 1.5, 0.7071, 1.4142]],
                    [[2.5, 1.5, 1., 1.], [2.5, 1.5, 0.7071, 1.4142]],
                ],
            ]))


    def test_base_size(self):
        # base_size == 1
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=1.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 1., 1.]],
                    [[1.5, 0.5, 1., 1.]],
                    [[2.5, 0.5, 1., 1.]],
                ],
                [
                    [[0.5, 1.5, 1., 1.]],
                    [[1.5, 1.5, 1., 1.]],
                    [[2.5, 1.5, 1., 1.]],
                ],
            ]))

        # base_size == 2
        anchors = generate_anchor_boxes(height=2,
                                        width=3,
                                        downsampling_factor=1,
                                        aspect_ratios=[1.],
                                        scales=[1.],
                                        base_size=2.)

        assert torch.allclose(anchors, torch.Tensor([
                [
                    [[0.5, 0.5, 2., 2.]],
                    [[1.5, 0.5, 2., 2.]],
                    [[2.5, 0.5, 2., 2.]],
                ],
                [
                    [[0.5, 1.5, 2., 2.]],
                    [[1.5, 1.5, 2., 2.]],
                    [[2.5, 1.5, 2., 2.]],
                ],
            ]))

