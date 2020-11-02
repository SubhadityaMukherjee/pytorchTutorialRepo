import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from torch.autograd import Variable
import torchsnooper
from torchvision.utils import save_image

Tensor = torch.cuda.FloatTensor 

# Boundary seeking loss [Ref](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/bgan/bgan.py

# This is new
def boundary_seeking_loss(y_pred, y_true): 
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2) + 1.0

def train(args, device, train_loader, epoch, netD, netG,nz, ndf, nc, optimizerD,optimizerG, batches_done):
    device = torch.device("cuda") # Sending to GPU
    discriminator_loss = torch.nn.BCELoss().to(device)

    for i, (imgs, _) in tqdm(enumerate(train_loader), 0):
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False) 
        real_imgs = Variable(imgs.type(Tensor))
        
        # Generator
        optimizerG.zero_grad()
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100)))) # 100 -> Latent
        gen_imgs = netG(z) # Batch
    
        g_loss = boundary_seeking_loss(netD(gen_imgs), valid)
        print(g_loss)
        g_loss.backward()
        optimizerG.step()

        # Discriminator

        optimizerD.zero_grad()
        real_loss = discriminator_loss(netD(real_imgs), valid)
        fake_loss = discriminator_loss(netD(gen_imgs.detach()), fake)
        # fake_loss = 0
        d_loss = (real_loss + fake_loss) /2
        d_loss.backward()
        optimizerD.step()


        print(
            f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}"
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.log_interval ==0:                 
            save_image(gen_imgs.data[:25], f"outputs/{batches_done}.png")
