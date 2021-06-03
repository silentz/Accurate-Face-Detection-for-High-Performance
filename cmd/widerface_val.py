"""
AInnoFace widerface val eval.

Usage:
    python3 cmd/widerface_val.py --checkpoint <checkpoint_path> --output <output_path>
"""

import os
import cv2
import torch
import argparse
import torchvision
import model.ainnoface
import model.widerface
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


def eval_model_on_image(ainnoface: model.ainnoface.AInnoFace, image: np.ndarray) -> list:
    bboxes = []
    scores = []
    result = []

    with torch.no_grad():
        model_input = [image,]
        _, ss, _ = ainnoface(model_input)
        proposals = ss.detach().cpu()[0]

    for bbox in proposals:
        x, y, w, h, p, _ = bbox
        p = torch.sigmoid(p)
        if p > 0.5:
            bboxes.append([x, y, w, h])
            scores.append(p)

    if len(bboxes) == 0:
        return []

    bboxes = torch.Tensor(bboxes)
    scores = torch.Tensor(scores)

    bboxes = torchvision.ops.box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
    nms_bboxes = torchvision.ops.nms(bboxes, scores, 0.4)
    res_bboxes = bboxes[nms_bboxes]
    res_scores = scores[nms_bboxes]
    res_bboxes = torchvision.ops.box_convert(res_bboxes, in_fmt='xyxy', out_fmt='xywh')

    for bbox, score in zip(res_bboxes, scores):
        x, y, w, h = bbox
        x, y, w, h, score = float(x), float(y), float(w), float(h), float(score)
        result.append([x, y, w, h, score])

    return result



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



def write_meta(meta: list, output_path):
    with open(output_path, 'w') as file:
        for item in meta:
            filename = item['filename']
            bboxes = item['bboxes']

            file.write(filename + '\n')
            file.write(str(len(bboxes)) + '\n')

            for bbox in bboxes:
                values = [str(x) for x in bbox]
                values = ' '.join(values)
                file.write(values + '\n')



def run_model(checkpoint: str, output_path: str):
    params = torch.load(checkpoint)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ainnoface = model.ainnoface.AInnoFace()
    ainnoface.load_state_dict(params)
    ainnoface = ainnoface.to(device)
    ainnoface.eval()

    meta = []
    dataset = model.widerface.WIDERFACEDataset(
            root='data/WIDER_val/images/',
            meta='data/wider_face_split/wider_face_val_bbx_gt.txt')

    for image in tqdm.tqdm(dataset):
        pixels = image.pixels(format='numpy')
        pixels = pad_image_if_needed(pixels, multiple_of=32)
        result = eval_model_on_image(ainnoface, pixels)

        meta.append({
                'filename': image.filename,
                'bboxes': result,
            })

        write_meta(meta, output_path)



def os_file(path: str):
    if os.path.isfile(path):
        return path

    raise ValueError('argument is not filesystem path')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=os_file)
    parser.add_argument('--output', required=False, type=str, default='output.txt')
    args = parser.parse_args()
    run_model(checkpoint=args.checkpoint, output_path=args.output)

