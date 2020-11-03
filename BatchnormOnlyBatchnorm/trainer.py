import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# This is new
def freezeOthers(m):
    for param in m.parameters():
        # check if batchnorm
        if (type(param) != 'torch.nn.modules.batchnorm.BatchNorm2d') or (type(param) != 'torch.nn.modules.batchnorm.BatchNorm1d'):
            param.grad[:] = 0 #set gradients to 0

def train(args, model, device, train_loader, optimizer, epoch):
    model.train() # Setting model to train
    device = torch.device("cuda") # Sending to GPU
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #Reset grads 

        output = model(data) # Passing batch through model

        loss = nn.CrossEntropyLoss()(output, target) # Will need to change everytime. Loss

        loss.backward() # Backprop
        optimizer.step() # Pass through optimizer
        model.apply(freezeOthers) # Dont train other layers

        if batch_idx % args.log_interval == 0:
            print(loss.item())
            if args.dry_run:
                break

