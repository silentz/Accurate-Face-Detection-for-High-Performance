"""
Train loop for AInnoFace model.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from model.loss import AInnoFaceLoss
from model.ainnoface import AInnoFace
from model.widerface import WIDERFACEDataset
from model.augment import AugmentedWIDERFACEDataset

# ===================== [CODE] =====================


class TrainModule(pl.LightningModule):

    def __init__(self):
        super(TrainModule, self).__init__()
        self._compute_fs = False
        self._device_ident = nn.Parameter(torch.empty(0))
        self.ainnoface = AInnoFace(backbone='resnet18', compute_first_step=self._compute_fs)
        self.loss = AInnoFaceLoss(two_step=self._compute_fs)


    def configure_optimizers(self):
        return torch.optim.SGD(self.ainnoface.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


    def training_step(self, batch, *args, **kwargs):
        images = [img.pixels(format='torch') for img in batch]
        bboxes = [img.torch_bboxes() for img in batch]
        bboxes = [img.to(self._device_ident.device) for img in bboxes]

        if self._compute_fs:
            fs, ss, anchors = self.ainnoface(images, device=self._device_ident.device)
            loss = self.loss(fs_proposal=fs, ss_proposal=ss, anchors=anchors, ground_truth=bboxes)
        else:
            ss, anchors = self.ainnoface(images, device=self._device_ident.device)
            loss = self.loss(ss_proposal=ss, anchors=anchors, ground_truth=bboxes)

        return loss


    #  def validation_step(self, batch, *args, **kwargs):
    #      images = [img.pixels(format='torch') for img in batch]
    #      bboxes = [img.torch_bboxes() for img in batch]

    #      fs, ss, anchors = self.ainnoface(images)
    #      loss = self.loss(fs, ss, anchors, bboxes)
    #      return loss



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


    #  def val_dataloader(self):
    #      dataset = WIDERFACEDataset(
    #              root='./data/WIDER_val/images/',
    #              meta='./data/wider_face_split/wider_face_val_bbx_gt.txt')
    #      return torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=1,
    #              collate_fn=custom_collate_fn)



def run_train_loop():
    pl.seed_everything(42)
    model = TrainModule()
    datamodule = WIDERFACEDatamodule()

    trainer = pl.Trainer(
            gpus=1,
            accumulate_grad_batches=2,
            max_epochs=10,
            precision=32,
        )

    trainer.fit(model=model, datamodule=datamodule)



if __name__ == '__main__':
    run_train_loop()

