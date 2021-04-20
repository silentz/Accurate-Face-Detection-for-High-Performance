"""
WIDER FACE dataset utilities. Includes loading utilities
for images and image annotations.
"""

# ==================== [IMPORT] ====================

import os
import copy
import typing
import collections

import cv2
import torch

# ===================== [CODE] =====================


class WIDERFACEImage:

    def __init__(self, filename: typing.Union[str, bytes, os.PathLike],
                       bboxes: typing.List[typing.Dict] = [],
                       lazy_load: bool = True):
        """
        Parameters
        ----------
        filename
            Path to image.

        bboxes
            List containing image bounding boxes information. Each element
            is a dictionary with following format:
                {
                    'x': 10,
                    'y': 20,
                    'w': 100,
                    'h': 120,
                    'blur': 0,
                    'expression': 0,
                    'illumination': 1,
                    'invalid': 0,
                    'occlusion': 0,
                    'pose': 0,
                }

        lazy_load
            If `True`, load image only when its pixels are appealed.
            Otherwise, load image into RAM immediately.
        """

        self.filename = copy.deepcopy(filename)
        self.bboxes = copy.deepcopy(bboxes)
        self._pixels = None

        if not lazy_load:
            self.pixels = self._load_image(self.filename)


    @property
    def pixels(self) -> torch.Tensor:
        """
        Get image pixels as `torch.Tensor`. Image pixels are not
        normilized (each pixels is a tuple of three 0-255 integers,
        each represents rgb channel value).
        """

        if self._pixels is None:
            self._pixels = self._load_image(self.filename)

        return self._pixels


    def _load_image(self, filename: typing.Union[str, bytes, os.PathLike]) -> torch.Tensor:
        pass



class WIDERFACEDataset(torch.utils.data.Dataset):
    """
    WIDER FACE dataset. Each item is `WIDERFACEImage` object, which contains image,
    its bounding boxes and its metainformation (expression, illumination, invalid,
    occlusion, pose).
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
            If `True`, __getitem__will return "heavy" `WIDERFACEImage` objects with image pixels
            loaded into RAM. Otherwise, __getitem__ will return "light" `WIDERFACEImage` objects
            without loading image pixels into RAM (loading performs automaticly when pixels are
            appealed).
        """

        if not os.path.exists(root):
            raise ValueError('Root is invalid: directory does not exist')

        if not os.path.exists(meta):
            raise ValueError('Metafile is invalid: file does not exist')

        if not isinstance(lazy_load_images, bool):
            raise ValueError('Lazy load option invalid: not bool')

        self._root_dir = copy.deepcopy(root)
        self._metafile = copy.deepcopy(meta)
        self._lazy_load_images = lazy_load_images
        self._images = self._read_metafile(meta, lazy_load_images)


    def _read_metafile(self, meta: typing.Union[str, bytes, os.PathLike],
                             lazy_load_images: bool) -> collections.OrderedDict:
        """
        Iterate through metafile and prepare `WIDERFACEImage` objects.
        See __init__ docstring for parameter description.

        Returns
        -------
        `collections.OrderedDict` object containing image mapping from filename
        info `WIDERFACEImage` objects.
        """

        result = collections.OrderedDict()
        current_line = 0
        total_lines = 0

        bbox_keys = ['x', 'y', 'w', 'h', 'blur', 'expression',
                        'illumination', 'invalid', 'occlusion', 'pose']

        with open(meta, 'r') as file:
            metafile_lines = [x.strip() for x in file.readlines()]
            total_lines = len(metafile_lines)

        while current_line < total_lines:
            filename = metafile_lines[current_line]
            filename = os.path.join(self._root_dir, filename)
            bbox_clean = []

            try:
                bbox_count = metafile_lines[current_line + 1]
            except:
                raise SyntaxError(f"Metafile format error: line {current_line+2} should contain bbox count")

            try:
                bbox_count = int(bbox_count)
            except:
                raise SyntaxError(f'Metafile format error: line {current_line+2} should be integer')

            try:
                bbox_raw = metafile_lines[current_line + 2:current_line + 2 + bbox_count]
            except:
                raise SyntaxError(f"Metafile format error: lines {current_line+3}-{current_line+3+bbox_count}" \
                                    " should contain bbox descriptions")

            try:
                for bbox_id in range(bbox_count):
                    bbox = [int(x) for x in bbox_raw[bbox_id].split()]
                    bbox = {x: y for x, y in zip(bbox_keys, bbox)}
                    bbox_clean.append(bbox)
            except:
                raise SyntaxError(f'Metafile syntax error: line {current_line+3+bbox_id}' \
                                    ' should contain only integers')

            result[filename] = WIDERFACEImage(filename=filename, bboxes=bbox_clean,
                                                lazy_load=lazy_load_images)
            current_line += 2 + bbox_count

        return result


    def __getitem__(self, index: typing.Union[int, str, bytes, os.PathLike]) -> WIDERFACEImage:
        pass


    def __len__(self) -> int:
        pass
