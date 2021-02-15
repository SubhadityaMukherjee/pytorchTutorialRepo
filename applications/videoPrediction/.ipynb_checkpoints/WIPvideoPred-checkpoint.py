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
import torchsnooper as sn
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import cv2
import json
from collections import Counter
import pickle
import numpy as np
from tqdm import tqdm_notebook

from MovingMNIST import MovingMNIST

from efficientnet_pytorch import EfficientNet

import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.utils.prune as prune

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# -

import torchsnooper as tsp


# + hide_output=false run_control={"marked": true}
# @tsp.snoop()
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

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

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size,
                            self.hidden_dim,
                            height,
                            width,
                            device=self.conv.weight.device),
                torch.zeros(batch_size,
                            self.hidden_dim,
                            height,
                            width,
                            device=self.conv.weight.device))


# + hide_output=false run_control={"marked": true}
# @tsp.snoop()
class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()
        """ ARCHITECTURE 
        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model
        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=nf,  # nf + 1
            hidden_dim=nf,
            kernel_size=(3, 3),
            bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3,
                    c_t3, h_t4, c_t4):

        outputs = []

        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :],
                cur_state=[h_t,
                           c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t,
                cur_state=[h_t2,
                           c_t2])  # we could concat to provide skip conn here

        encoder_vector = h_t2

        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(
                input_tensor=encoder_vector,
                cur_state=[h_t3,
                           c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(
                input_tensor=h_t3,
                cur_state=[h_t4,
                           c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        b, seq_len, _, h, w = x.size()

        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b,
                                                       image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b,
                                                         image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b,
                                                         image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b,
                                                         image_size=(h, w))

        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2,
                                   c_t2, h_t3, c_t3, h_t4, c_t4)

        #         print(outputs)
        return outputs


# + hide_output=false run_control={"marked": true}
# @tsp.snoop()
class LitModel(pl.LightningModule):
    def __init__(self,
                 n_hidden_dims,
                 learning_rate=1e-4,
                 weight_decay=0.0001,
                 hparams=None,
                 batch_size=64):
        super(LitModel, self).__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10
        self.n_hidden_dims = n_hidden_dims
        self.conv_lstm_model = EncoderDecoderConvLSTM(nf=self.n_hidden_dims,
                                                      in_chan=1)

    def create_video(self, x, y_hat, y):

        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat(
            [zeros.cpu().unsqueeze(0),
             difference.unsqueeze(0).cpu()], dim=1)[0].unsqueeze(1)

        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        grid = torchvision.utils.make_grid(final_image,
                                           nrow=self.n_steps_past +
                                           self.n_steps_ahead)

        return grid

#     @tsp.snoop()

    def forward(self, x):
        output = self.conv_lstm_model(x, future_seq=self.n_steps_ahead)
        return output

#     @tsp.snoop()

    def configure_optimizers(self):
        #         print(list(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=2,
                                                    gamma=0.1)

        return ([optimizer], [scheduler])


#     @tsp.snoop()

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.
                     n_steps_past, :, :, :], batch[:, self.
                                                   n_steps_past:, :, :, :],
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        preds = self(x).squeeze()
        loss = self.criterion(preds, y)
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()
        final_image = self.create_video(x, preds, y)
#         plt.plot(final_image)
#         plt.show()
#         plt.savefig(f"./{batch_idx}_im.jpg")
        self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
        plt.close()

        self.log('train_loss', loss)
        self.log('lr_saved', self.learning_rate)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.
                     n_steps_past, :, :, :], batch[:, self.
                                                   n_steps_past:, :, :, :],
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        preds = self(x).squeeze()
        loss = self.criterion(preds, y)
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()
        self.log('val_loss', loss)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['validation_loss'] for x in outputs]).mean()
        self.log('mean_val_loss', avg_loss)


# -

# # Load data

# + hide_output=false run_control={"marked": true}
class ImDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 data_dir: str = "/media/hdd/Datasets/movingMNIST/",
                 img_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10

    def setup(self, stage=None):

        self.train_dataset = MovingMNIST(train=True,
                                         data_root=self.data_dir,
                                         seq_len=self.n_steps_past +
                                         self.n_steps_ahead,
                                         image_size=64,
                                         num_digits=2)

        self.valid_dataset = MovingMNIST(train=False,
                                         data_root=self.data_dir,
                                         seq_len=self.n_steps_past +
                                         self.n_steps_ahead,
                                         image_size=64,
                                         num_digits=2)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=12,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=12)


# -

batch_size = 20
img_size = 128
n_hidden_dims = 64

dm = ImDataModule(batch_size=batch_size, img_size=img_size)
class_ids = dm.setup()

# + hide_output=false run_control={"marked": true}
# # Logs

model = LitModel(batch_size=batch_size, n_hidden_dims=n_hidden_dims)
# logger = CSVLogger("logs", name="lstmEncDec")
logger = TensorBoardLogger(save_dir = "logs")

# + hide_output=false run_control={"marked": true}
trainer = pl.Trainer(auto_select_gpus=True,
                     gpus=1,
                     precision=16,
                     profiler=True,
                     max_epochs=1,
                     callbacks=[pl.callbacks.ProgressBar()],
                     automatic_optimization=True,
                     enable_pl_optimizer=True,
                     accumulate_grad_batches=16,
                     logger=logger,
                     accelerator = 'ddp',
                     plugins = 'ddp_sharded')

# + hide_output=false run_control={"marked": true}
trainer.fit(model, dm)
# -



# + run_control={"marked": true}
trainer.test()

trainer.save_checkpoint('model1.ckpt')
# -

# # Inference

# +
# best_checkpoints = trainer.checkpoint_callback.best_model_path

# pre_model = LitModel.load_from_checkpoint(
#     checkpoint_path=best_checkpoints).to("cuda")

# pre_model.eval()
# pre_model.freeze()

# transforms = A.Compose([
#     A.CenterCrop(img_size, img_size, p=1.),
#     A.Resize(img_size, img_size),
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#                 max_pixel_value=255.0,
#                 p=1.0),
#     ToTensorV2(p=1.0),
# ],
#     p=1.)

# test_img = transforms(image=cv2.imread(
#     "/media/hdd/Datasets/asl/asl_alphabet_test/asl_alphabet_test/C_test.jpg"))

# y_hat = pre_model(test_img["image"].unsqueeze(0).to("cuda"))

# label_map

# label_map[int(torch.argmax(y_hat, dim=1))]
# -
















