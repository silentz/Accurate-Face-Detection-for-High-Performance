"""
Test suite for `model/widerface.py` module.
"""

# ==================== [IMPORT] ====================

import cv2
import torch
import pytest
import numpy as np
from PIL import Image

from ..model.widerface import (
    WIDERFACEImage,
    WIDERFACEDataset,
)

# ===================== [TEST] =====================


class TestWIDERFACEImage:

    def test_init(self):
        current_filename = './tests/test_dataset/mask.png'
        current_bboxes = [{'x': 1, 'y': 1, 'w': 1, 'h': 1}]

        # lazy load
        image = WIDERFACEImage(current_filename, current_bboxes, lazy_load=True)
        assert image.filename == current_filename
        assert image.bboxes == current_bboxes
        assert image.bboxes is not current_bboxes
        assert image._pixels is None

        # active load
        image = WIDERFACEImage(current_filename, current_bboxes, lazy_load=False)
        assert image.filename == current_filename
        assert image.bboxes == current_bboxes
        assert image.bboxes is not current_bboxes
        assert image._pixels is not None
        assert type(image._pixels) == np.ndarray


    def test_pixels(self):
        image_path = './tests/test_dataset/mask.png'
        image = WIDERFACEImage(image_path, [], True)
        image_pixels = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_pixels = image_pixels.astype(np.float32)

        assert torch.allclose(image.pixels(format='torch'), torch.from_numpy(image_pixels))
        assert np.allclose(image.pixels(format='numpy'), image_pixels)
        assert np.allclose(np.asarray(image.pixels(format='pillow')), image_pixels)

        with pytest.raises(ValueError):
            image.pixels(format='format_does_not_exist')


    def test_render(self):
        # bbox green w1
        bbox = {'x': 45, 'y': 45, 'w': 170, 'h': 150, 'color': (0, 255, 0), 'width': 1}
        image = WIDERFACEImage('./tests/test_dataset/mask.png', [bbox], True)
        image_pixels = cv2.imread('./tests/test_dataset/bbox_green_w1.png')
        image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2RGB)
        image_pixels = image_pixels.astype(np.float32)

        assert torch.allclose(image.render(format='torch'), torch.from_numpy(image_pixels))
        assert np.allclose(image.render(format='numpy'), image_pixels)
        assert np.allclose(np.asarray(image.render(format='pillow')), image_pixels)

        # bbox red w1
        bbox = {'x': 45, 'y': 45, 'w': 170, 'h': 150, 'color': (255, 0, 0), 'width': 1}
        image = WIDERFACEImage('./tests/test_dataset/mask.png', [bbox], True)
        image_pixels = cv2.imread('./tests/test_dataset/bbox_red_w1.png')
        image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2RGB)
        image_pixels = image_pixels.astype(np.float32)

        assert torch.allclose(image.render(format='torch'), torch.from_numpy(image_pixels))
        assert np.allclose(image.render(format='numpy'), image_pixels)
        assert np.allclose(np.asarray(image.render(format='pillow')), image_pixels)

        # bbox green w2
        bbox = {'x': 45, 'y': 45, 'w': 170, 'h': 150, 'color': (0, 255, 0), 'width': 2}
        image = WIDERFACEImage('./tests/test_dataset/mask.png', [bbox], True)
        image_pixels = cv2.imread('./tests/test_dataset/bbox_green_w2.png')
        image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2RGB)
        image_pixels = image_pixels.astype(np.float32)

        assert torch.allclose(image.render(format='torch'), torch.from_numpy(image_pixels))
        assert np.allclose(image.render(format='numpy'), image_pixels)
        assert np.allclose(np.asarray(image.render(format='pillow')), image_pixels)

        # check exception
        with pytest.raises(ValueError):
            image.render(format='format_does_not_exist')


    def test_add_bbox(self):
        image = WIDERFACEImage('tests/test_dataset/mask.png')
        image.add_bbox(45, 45, 170, 150, color=(0, 255, 0), width=1)

        target_image = cv2.imread('./tests/test_dataset/bbox_green_w1.png')
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image = target_image.astype(np.float32)

        assert np.allclose(image.render(format='numpy'), target_image)



class TestWIDERFACEDataset:
    pass

