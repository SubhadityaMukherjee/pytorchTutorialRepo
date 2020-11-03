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



FloatTensor = torch.cuda.FloatTensor

# This is new
LongTensor = torch.cuda.LongTensor

def sample_image(n_row, batches_done, netG):
    #Saves a grid of generated digits ranging from 0 to n_classes
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = netG(z, labels)
    save_image(gen_imgs.data, "outputs/%d.png" % batches_done, nrow=n_row, normalize=True)

# @torchsnooper.snoop()
def train(args, device, train_loader, epoch, netD, netG,nz, ndf, nc, optimizerD,optimizerG, batches_done, num_classes):
    device = torch.device("cuda") # Sending to GPU
    adversarial_loss =torch.nn.MSELoss() 
    for i, (imgs, labels) in tqdm(enumerate(train_loader), 0):
        batch_size = imgs.shape[0]
        # This is new : most of the things here
        
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad= False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad= False)

        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # Generator
        optimizerG.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size,100))))
        gen_labels = Variable(LongTensor(np.random.randint(0, num_classes, batch_size)))
        gen_imgs = netG(z, gen_labels)
        validity = netD(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        
        g_loss.backward()
        optimizerG.step()

        # Discriminator
        optimizerD.zero_grad()
        validity_real = netD(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        
        validity_fake = netD(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        d_loss = (d_real_loss + d_fake_loss) /2
        d_loss.backward()
        optimizerD.step()

        print(
            f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}"
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.log_interval ==0:                 
            sample_image(10, batches_done, netG)
