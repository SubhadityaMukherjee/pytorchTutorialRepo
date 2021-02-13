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

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from sklearn import model_selection, preprocessing
import numpy as np
from pytorch_lightning import Trainer
import torchvision.models as models
from collections import Counter
from pytorch_lightning import loggers as pl_loggers
import pickle
import os
import pandas as pd
os.environ["TORCH_HOME"] = "~/hdd/Datasets"

import torchsnooper as tp

# # Verifying the data

data_path = "/home/eragon/hdd/Datasets/movielens/"

df = pd.read_csv(data_path+"ratings.dat", delimiter = "::", names = ["user", "movie", "rating", "id"], engine = "python")

df.head(3)

df.movie.nunique()

df.user.nunique()

df.shape


# # Create model

class LitModel(pl.LightningModule):
    def __init__(self, num_users, num_movies, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.user_embed = nn.Embedding(num_users, 32)
        self.movie_embed = nn.Embedding(num_movies, 32)
        self.out = nn.Linear(64, 1)
        self.er = pl.metrics.MeanSquaredError()
        self.accuracy = pl.metrics.Accuracy()

    # will be used during inference
#     @tp.snoop()
    def forward(self, users, movies, ratings = None):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim = 1)
        output = self.out(output)
        return output
#     @tp.snoop()
    def training_step(self, train_batch, batch_idx):
        u, m, r = train_batch['users'] , train_batch['movies'],train_batch['ratings']
        logits = self.forward(u, m,r)
        r = r.view(-1,1)
        loss = nn.MSELoss()(logits,r)
        self.log('train_acc_step', self.er(r,logits))
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        u, m, r = test_batch['users'] , test_batch['movies'],test_batch['ratings']
        logits = self.forward(u, m,r)
        r = r.view(-1,1)
        loss = nn.MSELoss()(logits, r)
        self.log('test_acc_step', self.er(r,logits))
        self.log('test_loss', loss)
        return self.er(r,logits)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # Load data

# +
# "user", "movie", "rating", "id"
# -

class MovieDs:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]
        
        return {
            "users": torch.tensor(user, dtype = torch.long),
            "movies": torch.tensor(movie, dtype = torch.long),
            "ratings": torch.tensor(rating, dtype = torch.float),
        }


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size,data_dir: str = "/home/eragon/hdd/Datasets/movielens/"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # build dataset
        df = pd.read_csv(self.data_dir+"ratings.dat", delimiter = "::", names = ["user", "movie", "rating", "id"], engine = "python")
        lbl_user = preprocessing.LabelEncoder()
        lbl_movie = preprocessing.LabelEncoder()
        df.user = lbl_user.fit_transform(df.user.values)
        df.movie = lbl_movie.fit_transform(df.movie.values)
        print(df.dtypes)
        
        # split dataset
        self.train, self.test = model_selection.train_test_split(df, test_size = 0.1, random_state = 42, stratify = df.rating.values)
        print(len(self.train) , len(self.test))
        
    def train_dataloader(self):
        md =  MovieDs(users = self.train.user.values, movies = self.train.movie.values , ratings = self.train.rating.values)
        return DataLoader(md, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def test_dataloader(self):
        md =  MovieDs(users = self.test.user.values, movies = self.test.movie.values , ratings = self.test.rating.values)
        return DataLoader(md, batch_size=self.batch_size, num_workers=12)


dm = CSVDataModule(batch_size=1024)
dm.setup()

# # Model

model = LitModel(df.user.nunique() , df.movie.nunique())

trainer = pl.Trainer(auto_select_gpus=True, gpus=1,
                     precision=16, profiler=False,max_epochs=10,
                    callbacks = [pl.callbacks.ProgressBar()],
                     automatic_optimization=True,enable_pl_optimizer=True)

trainer.fit(model, dm)

trainer.test()

# +
# trainer.save_checkpoint('model1.ckpt')
# -








