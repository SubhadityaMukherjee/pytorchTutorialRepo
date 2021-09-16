# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + run_control={"marked": true}
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import cv2
import json
from collections import Counter, OrderedDict
import pickle
import numpy as np


import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from PIL import Image
import matplotlib.pyplot as plt

from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

# + run_control={"marked": true}
import torchsnooper as sn


# -

# # Create model

# + run_control={"marked": true}
class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            self._make_gen_block(
                latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0
            ),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Tanh(),
            )

        return gen_block

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.gen(noise)


# + run_control={"marked": true}
class Discriminator(nn.Module):
    def __init__(self, feature_maps: int, image_channels: int) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(
                feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True
            ),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)


# + run_control={"marked": true}
# Efficient net b5
# @sn.snoop()
class LitModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        weight_decay=0.0001,
        latent_dim=100,
        img_shape=(128, 128),
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.img_shape = (1, 32, 32)

        self.generator = self.getGenerator()
        self.discriminator = self.getDiscriminator()

        self.validation_z = torch.randn(8, self.latent_dim)
        self.eg_inp = torch.zeros(2, self.latent_dim)

    def getGenerator(self):
        gen = Generator(self.latent_dim, 64, 3)
        gen.apply(self._weights_init)
        return gen

    def getDiscriminator(self):
        disc = Discriminator(64, 3)
        disc.apply(self._weights_init)
        return disc

    def forward(self, x):
        x = x.view(*x.shape, 1, 1)
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat, y)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        optimizer_d = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler_g = torch.optim.lr_scheduler.StepLR(
            optimizer_g, step_size=2, gamma=0.1
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            optimizer_d, step_size=2, gamma=0.1
        )

        return ([optimizer_g, optimizer_d], [scheduler_g, scheduler_d])

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch

        result = None
        # disc
        if optimizer_idx == 0:
            res = self.discStep(real)

        # gen
        if optimizer_idx == 1:
            res = self.genStep(real)
        return res

    def genStep(self, real):
        gen_loss = self.genLoss(real)
        self.log("genLoss", gen_loss, on_epoch=True)
        return gen_loss

    def discStep(self, real):
        disc_loss = self.discLoss(real)
        self.log("discLoss", disc_loss, on_epoch=True)
        return disc_loss

    def discLoss(self, real):
        # train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.adversarial_loss(real_pred, real_gt)

        # train with fake
        fake_pred = self.getFakePred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.adversarial_loss(fake_pred, fake_gt)
        return real_loss + fake_loss

    def genLoss(self, real):
        fake_pred = self.getFakePred(real)
        fake_gt = torch.ones_like(fake_pred)
        fake_loss = self.adversarial_loss(fake_pred, fake_gt)
        return fake_loss

    def getFakePred(self, real):
        batch_size = len(real)
        noise = self.getNoise(batch_size, self.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)
        return fake_pred

    def getNoise(self, n_samples, latent_dim):
        return torch.randn(n_samples, latent_dim, device=torch.device("cuda"))

    def on_epoch_end(self):
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device="cuda")
        temp = (
            torchvision.utils.make_grid(
                self.generator(fixed_noise).detach().cpu(), padding=2, normalize=True
            )
            .detach()
            .cpu()
            .numpy()
        )
        temp = np.transpose(temp, (1, 2, 0))
        plt.imshow(temp)
        plt.savefig(f"./outputs/epoch{self.current_epoch}_output.png")


# -

# # Load data

# + run_control={"marked": true}
class ImageFolderDs(Dataset):
    def __init__(self, image_list, transforms=None):
        self.image_list = image_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image = Image.open(self.image_list[i])
        image = np.asarray(image, dtype=np.uint8)
        if self.transforms is not None:
            image = self.transforms(image=image)
        return image["image"]


# + run_control={"marked": true}
class ImDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, img_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose(
            [
                A.RandomResizedCrop(img_size, img_size, p=1.0),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                A.CoarseDropout(p=0.5),
                A.Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    def setup(self, stage=None):
        self.image_list = [self.data_dir + x for x in os.listdir(self.data_dir)]
        self.train_dataset = ImageFolderDs(
            image_list=self.image_list, transforms=self.train_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True
        )


# + run_control={"marked": true}
batch_size = 1000
img_size = 128
latent_dim = 100

# + run_control={"marked": true}
dm = ImDataModule(
    data_dir="/media/hdd/Datasets/celeba/img_align_celeba/",
    batch_size=batch_size,
    img_size=img_size,
)
class_ids = dm.setup()
# -

# # Logs

# + run_control={"marked": true}
model = LitModel(img_shape=(img_size, img_size, 3), latent_dim=latent_dim)

# + run_control={"marked": true}
logger = TensorBoardLogger(save_dir="logs")

# + run_control={"marked": true}
trainer = pl.Trainer(
    auto_select_gpus=True,
    gpus=1,
    precision=16,
    profiler=False,
    max_epochs=5,
    callbacks=[pl.callbacks.ProgressBar()],
    automatic_optimization=True,
    enable_pl_optimizer=True,
    logger=logger,
    accelerator="ddp",
    plugins="ddp_sharded",
)

# + run_control={"marked": true}
trainer.fit(model, dm)

# +
trainer.test()

trainer.save_checkpoint("model1.ckpt")
# -

# # Inference

best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(checkpoint_path=best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()

fixed_noise = torch.randn(64, latent_dim, 1, 1, device="cuda")
temp = (
    torchvision.utils.make_grid(
        pre_model.generator(fixed_noise).detach().cpu(), padding=2, normalize=True
    )
    .detach()
    .cpu()
    .numpy()
)
temp = np.transpose(temp, (1, 2, 0))
plt.imshow(temp)
