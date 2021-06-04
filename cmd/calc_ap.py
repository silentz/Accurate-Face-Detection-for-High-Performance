"""
Average precision metric.
"""

import os
import argparse

import torch
import torchvision
import model.widerface
import numpy as np

import sklearn.metrics
import matplotlib.pyplot as plt



def iou_scores(bboxes, gt_bboxes):
    bboxes = torch.Tensor(bboxes)
    gt_bboxes = torch.Tensor(gt_bboxes)
    bboxes = torchvision.ops.box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
    gt_bboxes = torchvision.ops.box_convert(gt_bboxes, in_fmt='xywh', out_fmt='xyxy')
    scores = torchvision.ops.box_iou(bboxes, gt_bboxes)
    scores = torch.max(scores, dim=1).values
    return scores.numpy()



def run_check(check_file: str):
    gt_df = model.widerface.WIDERFACEDataset(
            root='data/WIDER_val/images',
            meta='data/wider_face_split/wider_face_val_bbx_gt.txt')

    check_df = model.widerface.WIDERFACEDataset(
            root='./',
            meta=check_file)


    metrics = []
    pred = []
    labels = []

    for idx, image in enumerate(check_df):
        bboxes = [[b['x'], b['y'], b['w'], b['h']] for b in image.bboxes]
        scores = [b['blur'] for b in image.bboxes]

        gt_image = gt_df[image.filename]
        gt_bboxes = [[b['x'], b['y'], b['w'], b['h']] for b in gt_image.bboxes]

        if len(gt_bboxes) > len(bboxes):
            for _ in range(0, len(gt_bboxes) - len(bboxes)):
                metrics.append(0)
                pred.append(0)
                labels.append(1)

        if len(bboxes) == 0:
            continue

        iou = iou_scores(bboxes, gt_bboxes)
        for iou_idx, iou_score in enumerate(iou):
            metrics.append(scores[iou_idx])
            if iou_score >= 0.5:
                pred.append(1)
                labels.append(1)
            else:
                pred.append(1)
                labels.append(0)

    metrics = np.array(metrics)
    labels = np.array(labels)
    pred = np.array(pred)

    p, r, _ = sklearn.metrics.precision_recall_curve(labels, metrics)
    print('AP:', sklearn.metrics.auc(r, p))


def os_file(path: str):
    if os.path.isfile(path):
        return path

    raise ValueError('argument is not filesystem path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=os_file, help='file with predicted bounding boxes')
    args = parser.parse_args()
    run_check(args.check)

