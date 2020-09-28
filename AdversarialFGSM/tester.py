import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from Nets import *
from tqdm import tqdm

def fgsm_method(image, epsilon, data_grad): #defining the method 
    sign_data_grad = data_grad.sign() # get the signs of the gradients of the tensors
    attacked_image = image + epsilon*sign_data_grad # peturbation according to adding noise multiplied with a small number
    attacked_image = torch.clamp(attacked_image, 0, 1) # Stick within [0, 1]
    return attacked_image

device = "cuda:0"
model = Net().to(device)
model.load_state_dict(torch.load('./models/model.pt', map_location = 'cpu'))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/Desktop/Datasets/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True) # get test data

epsilons = [0, .05, .1, .15, .2, .25, .3] #defining epsilons


model.eval() # Setting model to test

def test(model, device, test_loader, epsilon):
    correct = 0
    adversarial_examplpes = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # We need to collect the gradients
        output = model(data)

        init_pred = output.max(1, keepdim = True)[1] #max log probabality
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data # Get data of our gradients

        attacked_data = fgsm_method(data, epsilon, data_grad) # attack!!

        output = model(attacked_data) # classify again

        final_pred = output.max(1, keepdim=True)[1]

        # Take a few ones which didnt work to plot
        if final_pred.item() == target.item():
            correct +=1
            if (epsilon == 0) and (len(adversarial_examplpes)<5):
                adv_ex = attacked_data.squeeze().detach().cpu().numpy()
                adversarial_examplpes.append( (init_pred.item(), final_pred.item(), adv_ex ))
        else:
            if len(adversarial_examplpes) <5:
                adv_ex = attacked_data.squeeze().detach().cpu().numpy()
                adversarial_examplpes.append( (init_pred.item(), final_pred.item(), adv_ex) )
    final_acc = correct/float(len(test_loader)) # Get accuracy

    print(f"Epsilon: {epsilon}\t Test accur = {correct}/{len(test_loader)} = {final_acc}")
    return final_acc, adversarial_examplpes

accuracies = []
examples = []

#Testing

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# Plot a few and save

cnt = 0
plt.figure(figsize = (8,10))

for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]),cnt)
        plt.xticks([],[])
        plt.yticks([],[])

        if j ==0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize = 14)
        orig, adv, ex = examples[i][j]
        plt.imshow(ex, cmap = "gray")

plt.tight_layout()
plt.savefig("./attacked.png")

