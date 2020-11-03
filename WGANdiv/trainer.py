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
# @torchsnooper.snoop()
# 
def train(args, device, train_loader, epoch, netD, netG,nz, ndf, nc, optimizerD,optimizerG, batches_done, k,p):
    device = torch.device("cuda") # Sending to GPU

    for i, (imgs, _) in tqdm(enumerate(train_loader), 0):
        real_imgs = Variable(imgs.type(Tensor), requires_grad = True)
        
        # Discriminator

        optimizerD.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100)))) # 100 -> Latent
        fake_imgs = netG(z) # Batch
        
        real_validity = netD(real_imgs) # real ims
        fake_validity = netD(fake_imgs) # fake im
         
        # This is new

        # Get Wdiv
        real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad= False)
        real_grad = autograd.grad(
            real_validity, real_imgs, real_grad_out, create_graph = True , retain_graph = True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1)**(p/2)

        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad= False)
        fake_grad = autograd.grad(
            fake_validity, fake_imgs, real_grad_out, create_graph = True , retain_graph = True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1)**(p/2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k/2
                
        
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
        d_loss.backward()

        optimizerD.step()
        optimizerG.zero_grad()

        # Gen train every n_critic steps

        if i % args.CRITIC_ITERS == 0:
            fake_imgs = netG(z)
            fake_validity = netD(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizerG.step()

            print(
                f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}"
            )

            if batches_done % args.log_interval ==0:                 
                save_image(fake_imgs.data[:25], f"outputs/{batches_done}.png")
        batches_done += args.CRITIC_ITERS
