import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # KL divergence

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Setting model to train
    device = torch.device("cuda")  # Sending to GPU
    train_loss = 0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()  # Reset grads
        recon_batch, mu, logvar = model(data)  # Passing batch through model

        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()  # Backprop
        train_loss += loss.item()
        optimizer.step()  # Pass through optimizer

        if batch_idx % args.log_interval == 0:
            print(loss.item())
            if args.dry_run:
                break
