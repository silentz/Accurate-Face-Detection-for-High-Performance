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
        self.ainnoface = AInnoFace(backbone='resnet18')
        self.loss = AInnoFaceLoss()


    def configure_optimizers(self):
        return torch.optim.SGD(self.ainnoface.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


    def training_step(self, batch, *args, **kwargs):
        images = [img.pixels(format='torch') for img in batch]
        bboxes = [img.torch_bboxes() for img in batch]
        bboxes = [img.to(self.device) for img in bboxes]

        fs, ss, anchors = self.ainnoface(images, device=self.device)
        loss = self.loss(fs, ss, anchors, bboxes)
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
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=6, num_workers=4,
                collate_fn=custom_collate_fn)


    def val_dataloader(self):
        dataset = WIDERFACEDataset(
                root='./data/WIDER_val/images/',
                meta='./data/wider_face_split/wider_face_val_bbx_gt.txt')
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=4,
                collate_fn=custom_collate_fn)



def run_train_loop():
    pl.seed_everything(42)
    model = TrainModule()
    datamodule = WIDERFACEDatamodule()

    trainer = pl.Trainer(
            gpus=2,
            accumulate_grad_batches=1,
            max_epochs=10,
            precision=16,
        )

    trainer.fit(model=model, datamodule=datamodule)



if __name__ == '__main__':
    run_train_loop()

