"""
WIDERFACE dataset implementation with augmentations.
"""

# ==================== [IMPORT] ====================

import cv2
import torch
import numpy as np
import albumentations as A

import os
import copy
import typing
from .widerface import WIDERFACEDataset, WIDERFACEImage


# ===================== [CODE] =====================


class AugmentedWIDERFACEDataset(WIDERFACEDataset):

    def __init__(self, root: typing.Union[str, bytes, os.PathLike],
                       meta: typing.Union[str, bytes, os.PathLike],
                       lazy_load: bool = True):
        """
        Parameters
        ----------
        root
            Root directory of dataset. Root dir is used to
            load dataset images from paths specified in metafile.

        meta
            Path to metafile of dataset. Metafile is a file
            containing metainformation about dataset images and
            bounding boxes. Format of metafile content:
                <image1 path>
                <number of bounding boxes for image1>
                x1, y1, w1, h1, blur, expression, illumination, invalid, occlusion, pose
                x2, y2, w2, h2, blur, expression, illumination, invalid, occlusion, pose
                ...
                <image2 path>
                <number of bounding boxes for image2>
                ...
            For more information of metafile content, see readme.txt file
            of WIDER FACE dataset annotation archive.

        lazy_load
            If `True`, __getitem__will return "heavy" `WIDERFACEImage` objects with image pixels
            loaded into RAM. Otherwise, __getitem__ will return "light" `WIDERFACEImage` objects
            without loading image pixels into RAM (loading performs automaticly when pixels are
            appealed).
        """

        super(AugmentedWIDERFACEDataset, self).__init__(root=root, meta=meta, lazy_load=lazy_load)

        self.photometric_distortions = A.Compose([
                A.OneOf([
                    A.OneOf([
                        A.Blur(blur_limit=5, p=0.3),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
                        A.RandomBrightness(limit=(-0.2, 0.2), p=0.2),
                    ]),
                    A.NoOp(),
                ])
            ])

        self.space_distortions = A.Compose([
                #  A.VerticalFlip(p=0.05),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=(-10, 10), p=0.1),
                A.PadIfNeeded(min_height=1024, min_width=1024, p=1, border_mode=cv2.BORDER_CONSTANT),
                A.CenterCrop(height=1024, width=1024, p=1),
                A.Resize(height=512, width=512, p=1),
            ], bbox_params=A.BboxParams(format='coco', min_visibility=0.1, label_fields=[]))


    def __getitem__(self, index: typing.Union[int, str, bytes, os.PathLike]) -> WIDERFACEImage:
        image = super().__getitem__(index)
        pixels = image.pixels(format='numpy')
        bboxes = [(bbox['x'], bbox['y'], bbox['w'], bbox['h']) for bbox in image.bboxes]

        # stage 1: photometric distortions
        result = self.photometric_distortions(image=pixels)
        pixels = result['image']

        # stage 2: space distortions
        result = self.space_distortions(image=pixels, bboxes=bboxes)
        pixels = result['image']
        bboxes = [{'x': r[0], 'y': r[1], 'w': r[2], 'h': r[3]} for r in result['bboxes']]

        return WIDERFACEImage(filename=image.filename, bboxes=copy.deepcopy(bboxes), pixels=pixels)

