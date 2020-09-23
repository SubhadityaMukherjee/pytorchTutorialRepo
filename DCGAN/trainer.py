import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import torch.optim as optim

# Loss part 1
criterion = nn.BCELoss()

def train(args, device, train_loader, epoch, netD, netG,
          real_label, fake_label, fixed_noise, nz, ngf, ndf, nc, optimizerD,
          optimizerG):
    device = torch.device("cuda") # Sending to GPU

    for batch_idx, data in tqdm(enumerate(train_loader), 0):
        # Update Discriminator, train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype = real_cpu.dtype,
                           device = device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake

        noise = torch.randn(batch_size, nz, 1, 1, device = device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4fD(G(z)):%.4f/%.4f' % (epoch, args.epochs, batch_idx, len(train_loader),errD.item(), errG.item(), D_x,D_G_z1, D_G_z2))
        # Save images 
        if batch_idx % 100 == 0:
            vutils.save_image(real_cpu,f"outputs/{str(batch_idx)}real_samples.png",
                             normalize = True)

            fake = netG(fixed_noise)
            
            vutils.save_image(fake.detach(),f"outputs/{str(batch_idx)}fake_samples.png",
                             normalize = True)
