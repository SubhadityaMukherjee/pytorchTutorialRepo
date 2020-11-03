# CGAN/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper

# 
# @torchsnooper.snoop()
class Generator(torch.nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(num_classes, num_classes)
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(100+num_classes, 128, normalize=False), # 100 -> latent dim ; This is new
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # This is new; Concat embedding of label + image -> input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes) # This is new
        
        self.model = nn.Sequential( # This is new  
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 1),
        )

    def forward(self, x, labels):
        img_flat = torch.cat((x.view(x.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(img_flat)
        return validity


# DCGAN/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf 
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf*8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input).view(-1,1).squeeze(1)



# BGAN/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This is new
class Generator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False), # 100 -> latent dim
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
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        img_flat = x.view(x.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# WGANgp/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False), # 100 -> latent dim
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
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        img_flat = x.view(x.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# draGAN/Nets.py
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
            nn.BatchNorm2d(128),
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
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.shape[0], -1)
        validity = self.adv_layer(img)
        return validity


# WGANdiv/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False), # 100 -> latent dim
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
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        img_flat = x.view(x.shape[0], -1)
        validity = self.model(img_flat)
        return validity


