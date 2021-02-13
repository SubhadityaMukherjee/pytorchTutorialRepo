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

# The dataset is a custom one -> has a few catgories of landscapes. (Replace with any data in folderwise. One for each class.)

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST, ImageFolder
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
import torchvision.models as models
from collections import Counter
from pytorch_lightning import loggers as pl_loggers
import pickle
import os
os.environ["TORCH_HOME"] = "~/Desktop/Datasets/"


# # Create model

class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.accuracy = pl.metrics.Accuracy()
#         self.average_p = pl.metrics.functional.average_precision()
#         self.val_accuracy = pl.metrics.Accuracy()
#         self.test_accuracy = pl.metrics.Accuracy()

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=True)
        # layers are frozen by using eval()
        self.feature_extractor.eval()

        n_sizes = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_sizes, num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self(x)
        self.log('train_acc_step', self.accuracy(preds, y))
        self.log('train_loss', loss)
#         self.log('avg_p_train', pl.metrics.functional.average_precision(preds, y,num_classes=self.num_classes))
        return loss
    

    def val_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self(x)
        self.log('val_acc_step', self.accuracy(preds, y))
#         self.log('avg_p_val', pl.metrics.functional.average_precision(preds, y,num_classes=self.num_classes))
        self.log('val_loss', loss)
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self(x)
        self.log('test_acc_step', self.accuracy(preds, y))
#         self.log('avg_p_val', pl.metrics.functional.average_precision(preds, y,num_classes=self.num_classes))
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # Load data

class ImDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_classes, data_dir: str = '/home/eragon/Desktop/Datasets/LandscapeSort/'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = num_classes

    def setup(self, stage=None):
        # build dataset
        im_dataset = ImageFolder(self.data_dir)
        # split dataset
        self.train, self.val, self.test = random_split(im_dataset, [80, 10, 9])
        self.train.dataset.transform = self.augmentation
        print(dict(Counter(self.train.dataset.targets)))
        print(self.train.dataset.class_to_idx)
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform
        return self.train.dataset.class_to_idx
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12)


dm = ImDataModule(batch_size=64,num_classes=5)
class_ids = dm.setup()

# Samples required by the custom ImagePredictionLogger callback to log image predictions.
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape

# # Logs

tensorboard --logdir ./lightning_logs

model = LitModel(( 3, 224, 224), 10)

tb_logger = pl_loggers.TensorBoardLogger('./lightning_logs/')

trainer = pl.Trainer(auto_select_gpus=True, gpus=1,
                     precision=16, profiler=False,max_epochs=100,
                    callbacks = [pl.callbacks.ProgressBar()],
                     automatic_optimization=True,logger = tb_logger,enable_pl_optimizer=True)

trainer.fit(model, dm)

trainer.test()

trainer.save_checkpoint('model1.ckpt')

with open("class_ids.pkl","wb+") as f:
    pickle.dump(class_ids, f)

# # Inference

from PIL import Image

with open("class_ids.pkl","rb+") as f:
    class_ids = pickle.load(f)

transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

class_ids = {v: k for k, v in class_ids.items()}
class_ids

m1 = LitModel.load_from_checkpoint('model1.ckpt')

t_path = "/home/eragon/Documents/ArtReferences/Landscapes/Desert/00000016_(6).jpg"

im_sh = Image.open(t_path).convert('RGB');im_sh.resize((128,128))

test = transform(im_sh).unsqueeze(0)

pred = class_ids[int(torch.argmax(m1(test), 1))];pred








