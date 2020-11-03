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
# This is new 
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



# @torchsnooper.snoop()
def train(args, device, train_loader, epoch, netD, netG,nz, ndf, nc, optimizerD,optimizerG, batches_done, lambda_gp):
    device = torch.device("cuda") # Sending to GPU

    for i, (imgs, _) in tqdm(enumerate(train_loader), 0):
        real_imgs = Variable(imgs.type(Tensor), requires_grad = True)
        
        # Discriminator

        optimizerD.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100)))) # 100 -> Latent
        fake_imgs = netG(z) # Batch
        
        real_validity = netD(real_imgs) # real ims
        fake_validity = netD(fake_imgs) # fake im
        gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp*gradient_penalty
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
