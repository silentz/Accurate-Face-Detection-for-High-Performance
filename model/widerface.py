"""
WIDER FACE dataset utilities. Includes loading utilities
for images and image annotations.
"""

# ==================== [IMPORT] ====================

import os
import copy
import typing
import pathlib
import collections

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage

# ===================== [CODE] =====================


class WIDERFACEImage:

    def __init__(self, filename: typing.Union[str, bytes, os.PathLike] = '',
                       bboxes: typing.List[typing.Dict] = [],
                       lazy_load: bool = True,
                       pixels: np.ndarray = None):
        """
        Parameters
        ----------
        filename
            Path to image.

        bboxes
            List containing image bounding boxes information. Each element
            is a dictionary with following format:
                {
                    'x': 10,                # required: int
                    'y': 20,                # required: int
                    'w': 100,               # required: int
                    'h': 120,               # required: int
                    'outline': (0, 255, 0), # optional: color of bbox
                    'width': 1,             # optional: bbox width
                    'label': '',            # optional: bbox label
                    ...                     # other info will not be used
                }

        pixels
            It is the way to create a WIDERFACEImage object using
            direct pixels assignment.

        lazy_load
            If `True`, load image only when its pixels are appealed.
            Otherwise, load image into RAM immediately.
        """

        self.filename = filename
        self.bboxes = copy.deepcopy(bboxes)
        self._pixels = copy.deepcopy(pixels)

        if (not lazy_load) and (self._pixels is None):
            self._pixels = self._load_image(self.filename)


    def pixels(self, format: str = 'torch') -> typing.Union[torch.Tensor, np.ndarray, PILImage]:
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


    def render(self, format: str = 'torch') -> typing.Union[torch.Tensor, np.ndarray, PILImage]:
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
        """

        image = self.pixels(format='pillow')
        draw = ImageDraw.Draw(image)

        for bbox in self.bboxes:
            x, y = bbox.get('x', None), bbox.get('y', None)
            w, h = bbox.get('w', None), bbox.get('h', None)
            outline = bbox.get('color', (0, 255, 0))
            width = bbox.get('width', 1)
            draw.rectangle([(x, y), (x + w, y + h)], fill=None, outline=outline, width=width)

        if format == 'pillow':
            return image

        if format == 'torch':
            return torch.from_numpy(np.array(image, dtype=np.uint8))

        if format == 'numpy':
            return np.array(image, dtype=np.uint8)

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
            image_pixels = image_pixels.astype(np.uint8)
            return image_pixels
        except:
            raise ValueError(f'Cannot load image: {filename}')


    def add_bbox(self, x: int, y: int, w: int, h: int,
                       color: int = (0, 255, 0),
                       width: int = 1,
                       label: str = '',
                       **kwargs):
        """
        Append new bbox to image object.

        Parameters
        ----------
        x, y, w, h
            Coordinates of left upper corner, width and height of
            bouding box.
        color
            Color of bounding box.
        width
            Width of bounding box outline.
        label
            Label of bounding box.
        """
        self.bboxes.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'color': color,
                'width': width,
                'label': label,
                **kwargs,
            })


    def torch_bboxes(self) -> torch.Tensor:
        result = []

        for bbox in self.bboxes:
            result.append(torch.Tensor([bbox['x'], bbox['y'], bbox['w'], bbox['h']]))

        if len(result) == 0:
            return torch.Tensor([])

        return torch.stack(result, dim=0)



class WIDERFACEDataset(torch.utils.data.Dataset):
    """
    WIDER FACE dataset. Each item is `WIDERFACEImage` object, which contains image,
    its bounding boxes and its metainformation (expression, illumination, invalid,
    occlusion, pose).
    """

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

        if not os.path.exists(root):
            raise ValueError('Root is invalid: directory does not exist')

        if not os.path.exists(meta):
            raise ValueError('Metafile is invalid: file does not exist')

        if not isinstance(lazy_load, bool):
            raise ValueError('Lazy load option invalid: not bool')

        self._root_dir = root
        self._metafile = meta
        self._lazy_load = lazy_load
        self._images, self._idx2key = self._read_metafile(meta, lazy_load)


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
            metafile_lines = [x for x in metafile_lines if len(x) > 0]
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

            # BUGFIX: there are lines in widerface metafile like this:
            # dir1/dir2/image.png
            # 0
            # 0 0 0 0 0 0 0 0 0 0
            # ...
            # next lines filter them out:
            if bbox_count == 0:
                if metafile_lines[current_line + 2] == '0 0 0 0 0 0 0 0 0 0':
                    current_line += 3
                    continue

            bbox_raw = metafile_lines[current_line + 2:current_line + 2 + bbox_count]

            try:
                for bbox_id in range(bbox_count):
                    bbox = [int(x) for x in bbox_raw[bbox_id].split()]
                    bbox = {x: y for x, y in zip(bbox_keys, bbox)}

                    if len(bbox) < 4:
                        raise SyntaxError(f'Metafile syntax error: line {current_line+3+bbox_id}' \
                                ' should contain at least 4 integers')

                    if bbox['w'] > 0 and bbox['h'] > 0:
                        bbox_clean.append(bbox)
            except SyntaxError as err:
                raise err
            except:
                raise SyntaxError(f'Metafile syntax error: line {current_line+3+bbox_id}' \
                                    ' should contain integers')

            file_path = pathlib.Path(filename)
            result[file_path] = WIDERFACEImage(filename=str(file_path), bboxes=bbox_clean,
                                                lazy_load=lazy_load_images)
            current_line += 2 + bbox_count

        idx2key = [x for x in result.keys()]
        return result, idx2key


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

        if isinstance(index, int):
            if index < 0:
                index = self._idx2key[len(self) + index]
            else:
                index = self._idx2key[index]

        file_path = pathlib.Path(index)
        return self._images[file_path]


    def __len__(self) -> int:
        """
        Returns count of images in the dataset.
        """

        return len(self._images)

