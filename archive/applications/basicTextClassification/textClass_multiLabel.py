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

# +
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from sklearn import model_selection, preprocessing
import numpy as np
import torchvision.models as models
from collections import Counter
import pickle
import os
import transformers
import pandas as pd

os.environ["TORCH_HOME"] = "~/hdd/Datasets"
# -

import torchsnooper as tp

# # Verifying the data

data_path = "/home/eragon/hdd/Datasets/moviereview/"

os.listdir(data_path)

df = pd.read_csv(data_path + "movie_data.csv", engine="python")

df.head(3)

df.review.nunique()

df.sentiment.nunique()

df.shape


# # Create model

class LitModel(pl.LightningModule):
    def __init__(self, num_classes, num_train_steps, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    #         self.er = pl.metrics.Accuracy()

    #     @tp.snoop()
    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        return x

    #     @tp.snoop()
    def training_step(self, train_batch, batch_idx):
        i, m, to, ta = (
            train_batch["ids"],
            train_batch["mask"],
            train_batch["token_type_ids"],
            train_batch["targets"],
        )
        logits = self.forward(i, m, to, ta)
        loss = nn.BCEWithLogitsLoss()(logits, ta.view(-1, 1))
        #         print(loss)
        #         acc = self.er(logits,ta.view(-1,1))
        #         self.log('train_acc_step', acc)
        self.log("train_loss", loss)
        return loss

    #         return acc , loss

    def test_step(self, test_batch, batch_idx):
        i, m, to, ta = (
            test_batch["ids"],
            test_batch["mask"],
            test_batch["token_type_ids"],
            test_batch["targets"],
        )
        logits = self.forward(i, m, to, ta)
        loss = nn.BCEWithLogitsLoss()(logits, ta.view(-1, 1))
        #         acc = self.er(logits,ta.view(-1,1))
        #         self.log('test_acc_step', acc)
        self.log("test_loss", loss)
        return loss

    #         return acc , loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # Load data

# +
# "user", "movie", "rating", "id"
# -

class SentiDs:
    def __init__(self, texts, targets, max_len=64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }
        return resp


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # build dataset
        df = pd.read_csv(self.data_dir + "movie_data.csv", engine="python")
        print(df.dtypes)

        # split dataset
        self.train, self.test = model_selection.train_test_split(
            df, test_size=0.2, random_state=42, stratify=df.sentiment.values
        )
        print(len(self.train), len(self.test))

    def train_dataloader(self):
        md = SentiDs(self.train.review.values, self.train.sentiment.values)
        return DataLoader(md, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def test_dataloader(self):
        md = SentiDs(self.test.review.values, self.test.sentiment.values)
        return DataLoader(md, batch_size=self.batch_size, num_workers=12)


def on_batch_end(self):
    if self.sched is not None:
        self.sched.step()


def plot_ima



# # Model

EPOCHS = 10
batch_size = 64
n_train_steps = int(len(df) / batch_size * EPOCHS)

dm = CSVDataModule(
    batch_size=batch_size, data_dir="/home/eragon/hdd/Datasets/moviereview/"
)
dm.setup()

model = LitModel(num_classes=1, num_train_steps=n_train_steps)

trainer = pl.Trainer(
    auto_select_gpus=True,
    gpus=1,
    precision=16,
    profiler=False,
    max_epochs=EPOCHS,
    callbacks=[pl.callbacks.ProgressBar()],
    automatic_optimization=True,
    enable_pl_optimizer=True,
)

trainer.fit(model, dm)

trainer.test()

# +
# trainer.save_checkpoint('model1.ckpt')
# -



# # Inference

# +
# model.load("model.ckpt", device = 'cuda')
# preds = model.predict(some_ds , device = "cuda")
# for p in preds:
#     print(p)
# -


