import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from Nets import *
from data import *
from PIL import *
from torchvision import transforms
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

prepr2 = transforms.Compose([
    transforms.ToPILImage(),
#    transforms.Resize(256),
    transforms.ToTensor()
])
model = torch.load("./models/model.pt", map_location="cuda:0")
print("Model loaded")
# send to eval

model.eval()
inputs = PIL.Image.open(
    "/home/eragon/Documents/datasets/bw2color/gray/Victorian1.png").convert("RGB")
print("Image loaded")
inputs = preprocess(inputs).float().unsqueeze_(0)
print(inputs.shape)
inputs = torch.autograd.Variable(inputs).to('cuda')
pred = model(inputs)
print(pred.shape)
pred = prepr2(pred.cpu()[0]).numpy()
print(pred.shape)
#print(pred)
plt.imshow(pred)
plt.savefig("./out.png")
print("Saved image")
