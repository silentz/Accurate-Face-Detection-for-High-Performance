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
import numpy as np
from PIL import Image, ImageDraw

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
                    'blur': 0,          # optional
                    'expression': 0,    # optional
                    'illumination': 1,  # optional
                    'invalid': 0,       # optional
                    'occlusion': 0,     # optional
                    'pose': 0,          # optional
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


    def pixels(self, format: str = 'torch') -> typing.Union[torch.Tensor, np.ndarray, Image]:
        """
        Get image pixels as `torch.Tensor`. Image pixels are not
        normilized (each pixels is a tuple of three 0-255 integers,
        each represents rgb channel value).

        Parameters
        ----------
        format
            Format used to store an image. Currently these
            formats are supported:
                * torch:  return image as torch.Tensor
                * numpy:  return image as numpy.ndarray
                * pillow: return image as PIL.Image
        """

        if self._pixels is None:
            self._pixels = self._load_image(self.filename)

        if format == 'torch':
            return torch.from_numpy(self._pixels)

        if format == 'numpy':
            return np.copy(self._pixels)

        if format == 'pillow':
            return Image.fromarray(np.uint8(self._pixels))

        raise ValueError(f"Unsupported image format: {format}")


    def render(self, format: str = 'torch',
                     outline: typing.Tuple[int, int, int] = (0, 255, 0),
                     width: int = 1) -> typing.Union[torch.Tensor, np.ndarray, Image]:
        """
        Get image pixels with applied bboxes as `torch.Tensor`. Image
        pixels are not normalized (each pixel is a tuple of three 0-255 integers,
        each represents rgb channel value).

        Parameters
        ----------
        format
            Format used to store an image. Currently these
            formats are supported:
                * torch:  return image as torch.Tensor
                * numpy:  return image as numpy.ndarray
                * pillow: return image as PIL.Image
        outline
            RGB color to use to paint bounding boxes.
        width
            Width of stroke to use to paint bounding boxes.
        """

        image = self.pixels(format='pillow')
        draw = ImageDraw.Draw(image)

        for bbox in self.bboxes:
            x, y = bbox.get('x', None), bbox.get('y', None)
            w, h = bbox.get('w', None), bbox.get('h', None)
            draw.rectangle([(x, y), (x + w, y + h)], fill=None, outline=outline, width=width)

        if format == 'pillow':
            return image

        if format == 'torch':
            return torch.from_numpy(np.array(image))

        if format == 'numpy':
            return np.array(image)

        raise ValueError(f"Unsupported image format: {format}")


    def _load_image(self, filename: typing.Union[str, bytes, os.PathLike]) -> np.ndarray:
        """
        Load image from file on filesystem into RAM.

        Parameters
        ----------
        filename
            Path to image.

        Returns
        -------
        numpy.ndarray containing non-normalized image pixels, each pixel is a tuple
        of three 0-255 integers, each represents rgb channel value.
        """

        try:
            image_pixels = cv2.imread(filename)
            image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2RGB)
            image_pixels = image_pixels.astype(np.float32)
            return image_pixels
        except:
            raise ValueError(f'Cannot load image: {filename}')



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
        """
        Get image of dataset.

        Parameters
        ----------
        index
            Path to an image from dataset or index.

        Returns
        -------
        `WIDERFACEImage` object representing required image.
        """

        return self._images[index]


    def __len__(self) -> int:
        """
        Returns count of images in the dataset.
        """

        return len(self._images)

