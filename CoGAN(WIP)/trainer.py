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

# This is new

@torchsnooper.snoop()
def train(args, device, train_loader,train_loader2, epoch, netD, netG,nz, ndf, nc, optimizerD,optimizerG, batches_done, num_classes):
    device = torch.device("cuda") # Sending to GPU
    adversarial_loss =torch.nn.MSELoss()
    for i, ((imgs1, _), (imgs2,_ )) in tqdm(enumerate(zip(train_loader, train_loader2))):
        batch_size = imgs1.shape[0]
        # This is new : most of the things here
        
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad= False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad= False)

        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0),3, ndf,ndf))
        imgs2 = Variable(imgs2.type(Tensor))

        # Generator
        optimizerG.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size,100))))
        gen_imgs1,gen_imgs2 = netG(z)
        print(gen_imgs1.size(), gen_imgs2.size())
        validity1, validity2 = netD(gen_imgs1, gen_imgs2)
        
        g_loss = (adversarial_loss(validity1, valid)+adversarial_loss(validity2, valid))/2
        
        g_loss.backward()
        optimizerG.step()

        # Discriminator
        optimizerD.zero_grad()
        validity1_real, validity2_real = netD(imgs1, imgs2) 
        validity1_fake, validity2_fake = netD(gen_imgs1.detach(), gen_imgs2.detach())
        d_loss = (
            adversarial_loss(validity1_real, valid)+
            adversarial_loss(validity1_fake, fake)+
            adversarial_loss(validity2_real, valid)+
            adversarial_loss(validity2_fake, fake)
            )/4

        d_loss.backward()
        optimizerD.step()

        print(
            f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}"
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.log_interval ==0:
            gen_images = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_images, f"outputs/{batches_done}.png", nrow= 8, normalize = True)
