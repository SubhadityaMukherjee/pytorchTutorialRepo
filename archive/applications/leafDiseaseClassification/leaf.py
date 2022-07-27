# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + run_control={"marked": false}
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import json
from collections import Counter
import pickle
import numpy as np

from efficientnet_pytorch import EfficientNet

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
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from sklearn import metrics, model_selection
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.integration import PyTorchLightningPruningCallback

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# -

import torchsnooper as sn

# # Look at data

main_path = "/media/hdd/Datasets/leafDisease/"

df = pd.read_csv(main_path + "train.csv")

df.head()

df.label.value_counts()

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
stratify = StratifiedKFold(n_splits=5)
for i, (t_idx, v_idx) in enumerate(
    stratify.split(X=df.image_id.values, y=df.label.values)
):
    df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)

with open(main_path + "label_num_to_disease_map.json", "r") as f:
    name_mapping = json.load(f)
name_mapping = {int(k): v for k, v in name_mapping.items()}
name_mapping

selected_images = []
fig = plt.figure(figsize=(16, 16))
for class_id, class_name in name_mapping.items():
    for i, (idx, row) in enumerate(
        df.loc[df["label"] == class_id].sample(4).iterrows()
    ):
        ax = fig.add_subplot(5, 4, class_id * 4 + i + 1, xticks=[], yticks=[])
        img = cv2.imread(f"{main_path}train_images/{row['image_id']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        ax.set_title(f"Image: {row['image_id']}. Label: {row['label']}")
        if i == 0:
            selected_images.append(img)


# # Create model

# Efficient net b5
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, weight_decay=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.enet = EfficientNet.from_pretrained(
            "efficientnet-b5", num_classes=self.num_classes
        )
        in_features = self.enet._fc.in_features
        self.enet._fc = nn.Linear(in_features, num_classes)

    #     @sn.snoop()

    def forward(self, x):
        out = self.enet(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        return ([optimizer], [scheduler])

    #     @sn.snoop()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log("train_acc_step", acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log("val_acc_step", acc)
        self.log("val_loss", loss)


class ImageClassDs(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]["image_id"])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        if self.train:
            y = self.df.iloc[index]["label"]
            return {
                "x": x,
                "y": y,
            }
        else:
            return {"x": x}

    def __len__(self):
        return len(self.df)


# # Load data

class ImDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        batch_size,
        num_classes,
        data_dir: str = "/media/hdd/Datasets/leafDisease/train_images/",
        img_size=(256, 256),
    ):
        super().__init__()
        self.df = df
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

        self.valid_transform = A.Compose(
            [
                A.CenterCrop(img_size, img_size, p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    def setup(self, stage=None):
        dfx = pd.read_csv("./train_folds.csv")
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]

        self.train_dataset = ImageClassDs(
            train, self.data_dir, train=True, transforms=self.train_transform
        )

        self.valid_dataset = ImageClassDs(
            val, self.data_dir, train=False, transforms=self.valid_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4)


batch_size = 64
num_classes = 5
img_size = 128

dm = ImDataModule(df, batch_size=batch_size, num_classes=num_classes, img_size=img_size)
class_ids = dm.setup()

# # Logs

model = LitModel(num_classes)

logger = CSVLogger("logs", name="eff-b5")

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

trainer.fit(model, dm)

# +
trainer.test()

trainer.save_checkpoint("model1.ckpt")
# -








