"""
Train loop for AInnoFace model.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from ..model.ainnoface import AInnoFace
from ..model.widerface import WIDERFACEDataset
from ..model.augment import AugmentedWIDERFACEDataset

# ===================== [CODE] =====================


class TrainModule(pl.LightningModule):

    def __init__(self):
        super(self, TrainModule).__init__()
        self.ainnoface = AInnoFace()


    def forward(self, batch):
        pass


    def configure_optimizers(self):
        pass


    def training_step(self, batch, batch_idx):
        pass


    def validation_step(self, batch, batch_idx):
        pass



class WIDERFACEDatamodule(pl.LightningDataModule):

    def __init__(self):
        super(self, WIDERFACEDatamodule).__init__()


    def train_dataloader(self):
        pass


    def val_dataloader(self):
        pass



def run_train_loop():
    pl.seed_everything(42)
    model = TrainModule()
    datamodule = WIDERFACEDatamodule()

    trainer = pl.Trainer(
            gpus=0,
            accumulate_grad_batches=1,
            max_epochs=10,
            precision=32,
        )

    trainer.fit(model=model, datamodule=datamodule)



if __name__ == '__main__':
    run_train_loop()

