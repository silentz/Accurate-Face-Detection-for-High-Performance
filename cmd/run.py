"""
AInnoFace model cli running tool.

Usage:
    python3 cmd/run.py --checkpoint <ckpt path> --image <image path> --output <output path>
"""

import os
import cv2
import torch
import argparse
import torchvision
import model.ainnoface
import model.widerface
import numpy as np


def pad_image_if_needed(pixels: np.ndarray, multiple_of: int) -> np.ndarray:
    height, width, channels = pixels.shape

    if not height % multiple_of == 0:
        pad_height = multiple_of - (height % multiple_of)
        pad = np.zeros(shape=(pad_height, width, channels), dtype=np.uint8)
        pixels = np.concatenate([pixels, pad], axis=0)

    height, width, channels = pixels.shape

    if not width % multiple_of == 0:
        pad_width = multiple_of - (width % multiple_of)
        pad = np.zeros(shape=(height, pad_width, channels), dtype=np.uint8)
        pixels = np.concatenate([pixels, pad], axis=1)

    return pixels



def run_model(checkpoint: str, image_path: str, output_path: str):
    params = torch.load(checkpoint)
    ainnoface = model.ainnoface.AInnoFace()
    ainnoface.load_state_dict(params)
    ainnoface.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = pad_image_if_needed(image, multiple_of=32)

    with torch.no_grad():
        model_input = [image,]
        fs, ss, anchors = ainnoface(model_input)

    ss = ss.detach().cpu()
    proposals = ss[0]

    result = model.widerface.WIDERFACEImage(pixels=image)
    candidates = []
    scores = []

    bboxes = proposals[:, 0:4]
    scores = torch.sigmoid(proposals[:, 4])
    bboxes = bboxes[scores >= 0.2]
    scores = scores[scores >= 0.2]

    candidates = torch.Tensor(bboxes)
    scores = torch.Tensor(scores)
    candidates = torchvision.ops.box_convert(candidates, in_fmt='xywh', out_fmt='xyxy')
    nms_boxes = torchvision.ops.nms(candidates, scores, 0.4)
    candidates = candidates[nms_boxes]
    nms_boxes = torchvision.ops.box_convert(candidates, in_fmt='xyxy', out_fmt='xywh')

    for bbox in nms_boxes:
        x, y, w, h = bbox
        result.add_bbox(x, y, w, h)

    final = result.render(format='pillow')
    final.save(output_path)



def os_file(path: str):
    if os.path.isfile(path):
        return path

    raise ValueError('argument is not filesystem path')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=os_file)
    parser.add_argument('--image', required=True, type=os_file)
    parser.add_argument('--output', required=False, type=str, default='output.png')
    args = parser.parse_args()
    run_model(checkpoint=args.checkpoint, image_path=args.image, output_path=args.output)

