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
import cv2
import json
from collections import Counter
import pickle
import numpy as np
import random


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

from PIL import Image
import PIL.ImageOps

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from sklearn import metrics, model_selection
from sklearn.model_selection import StratifiedKFold

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# -

import torchsnooper as sn


# # Look at data

# # Create model

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    def forward(self, output1,output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contr = torch.mean((1-label)*torch.pow(euclidean_distance,2)+(label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))
        return loss_contr


# +
# @sn.snoop()
class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=.1, weight_decay=0.0001, margin=1, img_size = 64):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.bilinear = True
        self.margin = margin
        self.criterion = ContrastiveLoss(margin = self.margin)
        self.img_size = img_size
        
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4,8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(8,8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*self.img_size*self.img_size, 500),
            nn.ReLU(inplace= True),
            
            nn.Linear(500, 500),
            nn.ReLU(inplace= True),
            
            nn.Linear(500, 5),
        )
    
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, inp1, inp2):
        output1 = self.forward_once(inp1)
        output2 = self.forward_once(inp2)
        return output1, output2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=2,
                                                    gamma=0.1)

        return ([optimizer], [scheduler])


#     @sn.snoop()

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1,output2, y)
        self.log('train_loss', loss,on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1,output2, y)
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('valid_loss',  avg_loss, on_step=False)
        return avg_loss 
    
    def on_batch_end(self):
        if self.sched is not None:
            self.sched.step()

# -

class ImageClassDs(Dataset):
    def __init__(self,
                 img_ds,
                 train: bool = True,
                 transforms=None, should_invert = None):
        self.img_ds = img_ds
        self.train = train
        self.transforms = transforms
        self.should_invert = should_invert
        
    def __getitem__(self, index):
        img0_tuple = random.choice(self.img_ds.imgs)
        should_get_same = random.randint(0,1)
        if should_get_same:
            while True:
                img1_tuple = random.choice(self.img_ds.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.img_ds.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0]).convert("L")
        img1 = Image.open(img1_tuple[0]).convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
        
        if self.transforms is not None:
            img0 = self.transforms(image = np.array(img0))['image']
            img1 = self.transforms(image = np.array(img1))['image']
        
        return img0, img1, torch.from_numpy(np.array([
            int(img1_tuple[1]!=img0_tuple[1])
        ],dtype = np.float32))

    def __len__(self):
        return len(self.img_ds.imgs)


# # Load data

class GrayToRGB(A.ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, p=1.0):
        super(GrayToRGB, self).__init__(p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
class RGBToGray(A.ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, p=1.0):
        super(RGBToGray, self).__init__(p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ImDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            data_dir,
            img_size=(256, 256)):
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose([
            GrayToRGB(),
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
            RGBToGray(),
            ToTensorV2(p=1.0),
        ],
            p=1.)

    def setup(self, stage=None):
        self.DatasetFolder = datasets.ImageFolder(self.data_dir)
        dataset = ImageClassDs(img_ds = self.DatasetFolder,transforms = self.train_transform)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        self.train_dataset, self.valid_dataset = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          pin_memory = True,
                          num_workers=12,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          pin_memory = True,
                          num_workers=12)


batch_size = 128
img_size = 100

dm = ImDataModule(batch_size=batch_size,
                  img_size=img_size,
                 data_dir = "/media/hdd/Datasets/celeba/")
class_ids = dm.setup()

# # Logs

model = LitModel(img_size=img_size)

logger = TensorBoardLogger(save_dir="logs")

trainer = pl.Trainer(auto_select_gpus=True,
                     gpus=1,
                     precision=16,
                     profiler=False,
                     max_epochs=1,
                     callbacks=[pl.callbacks.ProgressBar()],
                     enable_pl_optimizer=True,
                     logger=logger,
                     accumulate_grad_batches=16,
                     accelerator='ddp',
                     plugins='ddp_sharded')

trainer.fit(model, dm)

trainer.save_checkpoint('model1.ckpt')

# # Inference

# +
best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(checkpoint_path= best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()
# -

test_img = Image.open("/home/eragon/Downloads/t1.jpg")

y_hat = pre_model(transforms.ToTensor()(test_img).unsqueeze(0).to("cuda"))

transforms.ToPILImage()(y_hat.squeeze(0)).convert("RGB")


