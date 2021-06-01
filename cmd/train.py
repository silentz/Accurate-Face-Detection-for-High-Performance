"""
Train loop for AInnoFace model.
"""

# ==================== [IMPORT] ====================

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from model.loss import AInnoFaceLoss
from model.ainnoface import AInnoFace
from model.widerface import WIDERFACEDataset, WIDERFACEImage
from model.augment import AugmentedWIDERFACEDataset

warnings.filterwarnings('ignore')

# ===================== [CODE] =====================


class TrainModule(pl.LightningModule):

    def __init__(self):
        super(TrainModule, self).__init__()
        self._compute_fs = True
        self._device_ident = nn.Parameter(torch.empty(0))
        self.ainnoface = AInnoFace(backbone='resnet18')
        self.loss = AInnoFaceLoss()


    def configure_optimizers(self):
        #  return torch.optim.Adam(self.ainnoface.parameters())
        optimizer = torch.optim.SGD(params=self.ainnoface.parameters(),
                                    lr=0.0003125,
                                    momentum=0.9,
                                    weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    max_lr=0.01,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=403,
                    anneal_strategy='cos',
                    pct_start=0.03,
                    div_factor=32)
        return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
            }


    def training_step(self, batch, *args, **kwargs):
        images = [img.pixels(format='torch') for img in batch]
        bboxes = [img.torch_bboxes() for img in batch]
        bboxes = [img.to(self._device_ident.device) for img in bboxes]

        if self._compute_fs:
            fs, ss, anchors = self.ainnoface(images, device=self._device_ident.device)
            loss = self.loss(fs_proposal=fs, ss_proposal=ss, anchors=anchors, ground_truth=bboxes)
            self.logger.experiment.log_metric('train_loss', loss.item())
        else:
            ss, anchors = self.ainnoface(images, device=self._device_ident.device)
            loss = self.loss(ss_proposal=ss, anchors=anchors, ground_truth=bboxes)
            self.logger.experiment.log_metric('train_loss', loss.item())

        return loss


    def validation_step(self, batch, *args, **kwargs):
        images = [img.pixels(format='torch') for img in batch]
        bboxes = [img.torch_bboxes() for img in batch]
        gt_bboxes = [img.to(self._device_ident.device) for img in bboxes]

        if self._compute_fs:
            fs, ss, anchors = self.ainnoface(images, device=self._device_ident.device)
        else:
            ss, anchors = self.ainnoface(images, device=self._device_ident.device)

        result_images = []
        for idx, image in enumerate(batch):
            new_image = WIDERFACEImage(pixels=image.pixels(format='numpy'), filename='')
            image_proposals = ss[idx]
            bboxes = image_proposals[torch.sigmoid(image_proposals[:, 4]) >= 0.5][:, 0:4]
            for bbox in bboxes:
                new_image.add_bbox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
            for bbox in gt_bboxes[idx]:
                new_image.add_bbox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3], color=(255, 0, 0))
            rendered = new_image.render(format='pillow')
            self.logger.experiment.log_image('model_out', rendered, description=str(len(bboxes)))

        return 0



def custom_collate_fn(data):
    # separate function because pickle cannot use lambdas
    return data



class WIDERFACEDatamodule(pl.LightningDataModule):

    def __init__(self):
        super(WIDERFACEDatamodule, self).__init__()


    def train_dataloader(self):
        dataset = AugmentedWIDERFACEDataset(
                root='./data/WIDER_train/images/',
                meta='./data/wider_face_split/wider_face_train_bbx_gt.txt')
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=16, num_workers=1,
                collate_fn=custom_collate_fn)


    def val_dataloader(self):
        dataset = AugmentedWIDERFACEDataset(
                root='./data/WIDER_val/images/',
                meta='./data/wider_face_split/wider_face_val_bbx_gt.txt')
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=1,
                collate_fn=custom_collate_fn)



def run_train_loop():
    pl.seed_everything(42)
    model = TrainModule()
    datamodule = WIDERFACEDatamodule()

    neptune_logger = pl.loggers.NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project_name="silentz/AInnoFace",
            experiment_name='Neptune',
            params=dict(),
        )

    trainer = pl.Trainer(
            gpus=2,
            accumulate_grad_batches=2,
            logger=neptune_logger,
            val_check_interval=100,
            gradient_clip_val=0.5,
            limit_val_batches=2,
            max_epochs=100,
            precision=32,
        )

    trainer.fit(model=model, datamodule=datamodule)



if __name__ == '__main__':
    run_train_loop()

