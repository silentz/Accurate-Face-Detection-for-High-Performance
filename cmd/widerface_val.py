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



def process_proposals(proposals: torch.Tensor) -> list:
    bboxes = []
    scores = []
    result = []

    t_bboxes = proposals[:, 0:4]
    t_scores = torch.sigmoid(proposals[:, 4])
    t_bboxes = t_bboxes[t_scores > 0.5]
    t_scores = t_scores[t_scores > 0.5]

    for bbox, score in zip(t_bboxes, t_scores):
        x, y, w, h = bbox
        bboxes.append([x, y, w, h])
        scores.append(score)

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



def eval_model_on_image(ainnoface: model.ainnoface.AInnoFace,
                        image: np.ndarray,
                        device: torch.device) -> list:
    with torch.no_grad():
        model_input = [image,]
        _, ss, _ = ainnoface(model_input, device)
        proposals = ss.detach().cpu()[0]

    return process_proposals(proposals)


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



def pad_image(pixels: np.ndarray, h: int, w: int) -> np.ndarray:
    init_h, init_w, init_c = pixels.shape
    result = np.zeros(shape=(h, w, init_c), dtype=np.uint8)
    result[0:init_h, 0:init_w, 0:init_c] = pixels
    return result



def run_model(checkpoint: str, output_path: str):
    params = torch.load(checkpoint)

    device_count = torch.cuda.device_count()
    ainnoface = model.ainnoface.AInnoFace()
    ainnoface.load_state_dict(params)
    ainnoface.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ainnoface = ainnoface.to(device)

    meta = []
    dataset = model.widerface.WIDERFACEDataset(
            root='data/WIDER_val/images/',
            meta='data/wider_face_split/wider_face_val_bbx_gt.txt')

    filenames = []
    batch = []

    batch_size = 6
    length = len(dataset)
    idx = 0

    for image in tqdm.tqdm(dataset):
        pixels = image.pixels(format='numpy')

        if len(batch) < batch_size:
            batch.append(pixels)
            filenames.append(image.filename)

        if (len(batch) == batch_size) or (idx + 1 == length):
            max_height = 0
            max_width = 0

            for image in batch:
                height, width, _ = image.shape
                max_height = max(max_height, height)
                max_width = max(max_width, width)

            batch = [pad_image(x, h=max_height, w=max_width) for x in batch]
            batch = [pad_image_if_needed(x, multiple_of=32) for x in batch]
            batch = np.stack(batch, axis=0)

            with torch.no_grad():
                _, proposals, _ = ainnoface(batch, device)

            for idx in range(len(batch)):
                meta.append({
                        'filename': filenames[idx],
                        'bboxes': process_proposals(proposals[idx])
                    })

            write_meta(meta, output_path)
            filenames = []
            batch = []

        idx += 1



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

