import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This is new
class Generator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),  # 100 -> latent dim
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        img_flat = x.view(x.shape[0], -1)
        validity = self.model(img_flat)
        return validity


import hiddenlayer as hl
from torch.autograd import Variable

x = Variable(torch.rand(1, 3, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save("gp.png")
