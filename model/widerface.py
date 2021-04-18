"""
WIDER FACE dataset utilities. Includes loading utilities
for images and image annotations.
"""

# ==================== [IMPORT] ====================

import os
import copy
import torch
import typing

# ===================== [CODE] =====================


class WIDERFACEImage:

    def __init__(self):
        pass



class WIDERFACEDataset(torch.utils.data.Dataset):
    """
    WIDER FACE dataset. Each item is WIDERFACEImage object, which contains image,
    its bounding boxes and its metainformation (expression, illumination, invalid,
    occlusion, pose). Uses singleton image index, so loading and copying becomes
    faster.
    """

    def __init__(self, root: typing.Union[str, bytes, os.PathLike],
                       meta: typing.Union[str, bytes, os.PathLike],
                       lazy_load_images: bool = True):
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

        lazy_load_images
            If `True`, __getitem__will return "heavy" WIDERFACEImage objects with image pixels
            loaded into RAM. Otherwise, __getitem__ will return "light" WIDERFACEImage objects
            without loading image pixels into RAM (loading performs automaticly when pixels are
            appealed).
        """

        # check if root dir exists
        if not os.path.exists(root):
            raise ValueError('Root is invalid: directory does not exist')

        # check if metafile exists
        if not os.path.exists(meta):
            raise ValueError('Metafile is invalid: file does not exist')

        if not isinstance(lazy_load_images, bool):
            raise ValueError('Lazy load option invalid: not bool')

        self._root_dir = copy.deepcopy(root)
        self._metatile = copy.deepcopy(meta)
        self._lazy_load_images = lazy_load_images
        self._process_metafile()


    def _process_metafile(self):
        pass


    def __getitem__(self, index: typing.Union[int, str, bytes, os.PathLike]) -> WIDERFACEImage:
        pass


    def __len__(self) -> int:
        pass
