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
# @torchsnooper.snoop()
def compute_gradient_penalty(D, X, lambda_gp):
    # Random weight term for interpolation between real and fake samples
    device = torch.device("cuda")
    alpha = Tensor(np.random.random(size=X.shape)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = alpha * X + (
        (1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).to(device))
    )
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)
    fake = Variable(Tensor(X.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# @torchsnooper.snoop()
def train(
    args,
    device,
    train_loader,
    epoch,
    netD,
    netG,
    nz,
    ndf,
    nc,
    optimizerD,
    optimizerG,
    batches_done,
    lambda_gp,
):
    device = torch.device("cuda")  # Sending to GPU
    adversarial_loss = torch.nn.BCELoss().to(device)
    for i, (imgs, _) in tqdm(enumerate(train_loader), 0):
        imgs = imgs.to(device)
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        #  Train NetG

        optimizerG.zero_grad()

        # Sample noise as netG input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))

        # Generate a batch of images
        gen_imgs = netG(z)

        # Loss measures netG's ability to fool the netD
        g_loss = adversarial_loss(netD(gen_imgs), valid)

        g_loss.backward()
        optimizerG.step()

        #  Train NetD

        optimizerD.zero_grad()

        # Measure netD's ability to classify real from generated samples
        real_loss = adversarial_loss(netD(real_imgs), valid)
        fake_loss = adversarial_loss(netD(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Calculate gradient penalty
        gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, lambda_gp)
        gradient_penalty.backward()

        optimizerD.step()

        print(f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}")

        if batches_done % args.log_interval == 0:
            save_image(gen_imgs.data, f"outputs/{batches_done}.png", normalize=True)
        batches_done += args.CRITIC_ITERS
