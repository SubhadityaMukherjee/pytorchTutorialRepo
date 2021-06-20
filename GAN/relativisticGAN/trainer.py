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
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)  # New

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

        # New
        real_pred = netD(real_imgs).detach()
        fake_pred = netD(gen_imgs)

        if args.avg:
            g_loss = adversarial_loss(
                fake_pred - real_pred.mean(0, keepdim=True), valid
            )
        else:
            g_loss = adversarial_loss(fake_pred - real_pred, valid)

        g_loss.backward()
        optimizerG.step()

        #  Train NetD

        optimizerD.zero_grad()

        # New
        real_pred = netD(real_imgs)
        fake_pred = netD(gen_imgs.detach())
        if args.avg:
            real_loss = adversarial_loss(
                real_pred - fake_pred.mean(0, keepdim=True), valid
            )
            fake_loss = adversarial_loss(
                fake_pred - fake_pred.mean(0, keepdim=True), fake
            )
        else:
            real_loss = adversarial_loss(real_pred - fake_pred, valid)
            fake_loss = adversarial_loss(fake_pred - fake_pred, fake)

        d_loss = (real_loss + fake_loss) / 2
        optimizerD.step()

        print(f"D loss: {d_loss.item()} ; G loss: {g_loss.item()}")

        if batches_done % args.log_interval == 0:
            save_image(
                gen_imgs.data[:25], f"outputs/{batches_done}.png", normalize=True
            )
        batches_done += args.CRITIC_ITERS
