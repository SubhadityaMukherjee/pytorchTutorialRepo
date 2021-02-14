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

# + run_control={"marked": false}
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
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

from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.integration import PyTorchLightningPruningCallback
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# -

import torchsnooper as sn

# # Look at data 
# - Create a csv for easy loading

main_path = "/media/hdd/Datasets/asl/"

all_ims = glob.glob(main_path + "/*/*/*/*.jpg")
all_ims[0]

len(all_ims)


def create_label(x):
    return x.split("/")[-2]


df = pd.DataFrame.from_dict({x: create_label(x)
                             for x in all_ims},
                            orient='index').reset_index()
df.columns = ["image_id", "label"]

df.head()

temp = preprocessing.LabelEncoder()
df['label'] = temp.fit_transform(df.label.values)

label_map = {i: l for i, l in enumerate(temp.classes_)}

df.label.nunique()

df.label.value_counts()

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
stratify = StratifiedKFold(n_splits=5)
for i, (t_idx, v_idx) in enumerate(
        stratify.split(X=df.image_id.values, y=df.label.values)):
    df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)

pd.read_csv("train_folds.csv").head(1)


# # Create model

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cur + i * g
        h_next = o * torch.tanh(c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width),
            torch.zeros(batch_size, self.hidden_dim, height, width),
        )


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()
        
        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions
        
        self.encoder_1 = ConvLSTMCell(input_dim=in_chan,
                                      hidden_dim=nf,
                                      kernel_size=(3,3),
                                      bias=True)
        self.encoder_2 = ConvLSTMCell(input_dim=nf,
                                      hidden_dim=nf,
                                      kernel_size=(3,3),
                                      bias=True)
        self.decoder_1 = ConvLSTMCell(input_dim=nf,
                                      hidden_dim=nf,
                                      kernel_size=(3,3),
                                      bias=True)
        self.decoder_2 = ConvLSTMCell(input_dim=nf,
                                      hidden_dim=nf,
                                      kernel_size=(3,3),
                                      bias=True)
        self.decoder_CNN = nn.Conv3d(input_dim=nf,
                                      hidden_dim=1,
                                      kernel_size=(1,3,3),
                                      padding = (0,1,1))
    
    def autoencoder(self,x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        for t in range(seq_len):
            h_t, c_t
        





# +
# Efficient net b5
# @sn.snoop()
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, weight_decay=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.enet = EfficientNet.from_pretrained('efficientnet-b5',
                                                 num_classes=self.num_classes)
        in_features = self.enet._fc.in_features
        self.enet._fc = nn.Linear(in_features, num_classes)


#     @sn.snoop()

    def forward(self, x):
        out = self.enet(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=2,
                                                    gamma=0.1)

        return ([optimizer], [scheduler])

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log('train_acc_step', acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log('val_acc_step', acc)
        self.log('val_loss', loss)


# -

class ImageClassDs(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 imfolder: str,
                 train: bool = True,
                 transforms=None):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.df.iloc[index]['image_id']
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if (self.transforms):
            x = self.transforms(image=x)['image']

        y = self.df.iloc[index]['label']
        return {
            "x": x,
            "y": y,
        }

    def __len__(self):
        return len(self.df)


# # Load data

class ImDataModule(pl.LightningDataModule):
    def __init__(self,
                 df,
                 batch_size,
                 num_classes,
                 data_dir: str = "/media/hdd/Datasets/asl/",
                 img_size=(256, 256)):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, p=1.0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit=0.2,
                                 val_shift_limit=0.2,
                                 p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1),
                                       p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ],
            p=1.)

        self.valid_transform = A.Compose([
            A.CenterCrop(img_size, img_size, p=1.),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2(p=1.0),
        ],
            p=1.)

    def setup(self, stage=None):
        dfx = pd.read_csv("./train_folds.csv")
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]

        self.train_dataset = ImageClassDs(train,
                                          self.data_dir,
                                          train=True,
                                          transforms=self.train_transform)

        self.valid_dataset = ImageClassDs(val,
                                          self.data_dir,
                                          train=False,
                                          transforms=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=12,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=12)


batch_size = 64
num_classes = 29
img_size = 128

dm = ImDataModule(df,
                  batch_size=batch_size,
                  num_classes=num_classes,
                  img_size=img_size)
class_ids = dm.setup()

# # Logs

model = LitModel(num_classes)

logger = CSVLogger("logs", name="eff-b5")

trainer = pl.Trainer(auto_select_gpus=True,
                     gpus=1,
                     precision=16,
                     profiler=False,
                     max_epochs=5,
                     callbacks=[pl.callbacks.ProgressBar()],
                     automatic_optimization=True,
                     enable_pl_optimizer=True,
                     logger=logger,
                     accelerator='ddp',
                     plugins='ddp_sharded')

trainer.fit(model, dm)

# +
trainer.test()

trainer.save_checkpoint('model1.ckpt')
# -

# # Inference

best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(
    checkpoint_path=best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()

transforms = A.Compose([
    A.CenterCrop(img_size, img_size, p=1.),
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0),
    ToTensorV2(p=1.0),
],
    p=1.)

test_img = transforms(image=cv2.imread(
    "/media/hdd/Datasets/asl/asl_alphabet_test/asl_alphabet_test/C_test.jpg"))

y_hat = pre_model(test_img["image"].unsqueeze(0).to("cuda"))

label_map

label_map[int(torch.argmax(y_hat, dim=1))]
















