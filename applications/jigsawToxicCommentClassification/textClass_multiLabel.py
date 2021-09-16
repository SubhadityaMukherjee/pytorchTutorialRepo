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

# +
from typing import Tuple, List
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from sklearn import model_selection, preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np
import torchvision.models as models
from collections import Counter
import pickle
import os
import transformers
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, BertPreTrainedModel
import pandas as pd

os.environ["TORCH_HOME"] = "/media/hdd/Datasets"
# -

import torchsnooper as tp

# # Verifying the data

data_path = "/media/hdd/Datasets/jigsaw/"

os.listdir(data_path)

df = pd.read_csv(data_path + "train.csv", engine="python")

df.head(3)

df.shape


# # Create model

# +
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
        self.num_classes = num_classes
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        self.num_train_steps = num_train_steps
        self.acc = pl.metrics.PrecisionRecallCurve(num_classes=self.num_classes)

    #     @tp.snoop()
    def forward(self, ids, mask=None, token_type_ids=None):
        _, x = self.bert(ids, mask, token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

    #     @tp.snoop()

    def training_step(self, train_batch, batch_idx):
        #         i,m,to,ta= train_batch['ids'] , train_batch['mask'],train_batch['token_type_ids'], train_batch['targets']
        x, y = train_batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        #         print(loss)
        #         acc = self.er(logits,ta.view(-1,1))

        #         self.log('train_acc_step', self.acc())
        self.log("train_loss", loss)
        return loss

    #         return acc , loss

    def test_step(self, test_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        #         acc = self.er(logits,ta.view(-1,1))
        #         self.log('test_acc_step', acc)
        self.log("test_loss", loss)
        return loss

    #         return acc , loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# -

# # Load data

# +
# id 	comment_text 	toxic 	severe_toxic 	obscene 	threat 	insult 	identity_hate

# +
class SentiDs:
    def __init__(self, dataframe, max_len=64):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=False
        )
        # run once
        #         self.X = []
        #         self.Y = []
        #         for i, (row) in tqdm(dataframe.iterrows()):
        #             x, y = self.row_to_tensor(self.tokenizer,row)
        #             self.X.append(x)
        #             self.Y.append(y)

        #         with open("x_saved.pkl","wb+") as f:
        #             pickle.dump(self.X,f)

        #         with open("y_saved.pkl","wb+") as f:
        #             pickle.dump(self.Y,f)

        # comment above and run these next time

        with open("x_saved.pkl", "rb+") as f:
            self.X = pickle.load(f)

        with open("y_saved.pkl", "rb+") as f:
            self.Y = pickle.load(f)
        print("Loaded")

        self.max_len = max_len

    @staticmethod
    def row_to_tensor(
        tokenizer: BertTokenizer, row: pd.Series
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tokens = tokenizer.encode(row["comment_text"], add_special_tokens=True)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        y = torch.FloatTensor(
            row[
                [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                ]
            ]
        )
        return x, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -


def collate_fn(
    batch: List[Tuple[torch.LongTensor, torch.LongTensor]]
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # build dataset
        df = pd.read_csv(self.data_dir + "train.csv", engine="python")
        print(df.dtypes)

        # split dataset
        self.train, self.test = model_selection.train_test_split(
            df, test_size=0.2, random_state=42
        )
        print(len(self.train), len(self.test))

    def train_dataloader(self):
        md = SentiDs(self.train)
        train_sampler = RandomSampler(self.train)
        return DataLoader(
            md,
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

    def test_dataloader(self):
        md = SentiDs(self.test)
        test_sampler = RandomSampler(self.test)
        return DataLoader(
            md,
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=collate_fn,
            sampler=test_sampler,
        )


def on_batch_end(self):
    if self.sched is not None:
        self.sched.step()


# # Model

EPOCHS = 1
batch_size = 20
num_classes = 6
n_train_steps = int(len(df) / batch_size * EPOCHS)

dm = CSVDataModule(batch_size=batch_size, data_dir="/media/hdd/Datasets/jigsaw/")
dm.setup()

model = LitModel(num_classes=num_classes, num_train_steps=n_train_steps)

logger = pl_loggers.CSVLogger("logs", name="eff-b5")

trainer = pl.Trainer(
    auto_select_gpus=True,
    gpus=1,
    precision=16,
    profiler=False,
    max_epochs=EPOCHS,
    callbacks=[pl.callbacks.ProgressBar()],
    automatic_optimization=True,
    enable_pl_optimizer=True,
    accelerator="ddp",
    plugins="ddp_sharded",
    logger=logger,
)

trainer.fit(model, dm)

trainer.test()

trainer.save_checkpoint("model1.ckpt")


# # Inference

best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(checkpoint_path=best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()

tokenizer_inf = transformers.BertTokenizer.from_pretrained(
    "bert-base-u`ncased", do_lower_case=False
)

mapping = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
mapping_d = {i: mapping[i] for i in range(len(mapping))}

inp = "you are an idiot dude"
inp = tokenizer_inf.encode(inp, add_special_tokens=True)
print(inp)
pr = torch.Tensor(inp).unsqueeze(0).long()
pr
# print(tokenizer_inf.pad_token_id)
pr = pad_sequence(pr, batch_first=True, padding_value=tokenizer_inf.pad_token_id).to(
    "cuda"
)
output = pre_model(pr)
output

mapping_d[int(torch.argmax(output))]
