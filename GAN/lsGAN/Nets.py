import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, img_shape, args):
        super().__init__()
        self.img_shape = img_shape
        self.init_size = self.img_shape[1]//4
        self.l1 = nn.Sequential(nn.Linear(100, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        img = self.l1(x)
        img = img.view(img.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(img)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, img_shape, args):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.shape[0], -1)
        validity = self.adv_layer(img)
        return validity
