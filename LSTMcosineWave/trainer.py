import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tester import test
import matplotlib.pyplot as plt
import numpy as np

criterion = nn.MSELoss()

def train(args, model, device, data, input, target, test_input, test_target, optimizer, epoch):
    device = torch.device("cuda") # Sending to GPU

    def closure():
        optimizer.zero_grad() #Reset grads 
        out = model(input) # Passing batch through model

        loss = criterion(out, target)
        print(f"Loss: {loss.item()}")
        loss.backward() # Backprop
        return loss

    optimizer.step(closure) # Pass through optimizer

    y, future = test(model, test_input, test_target)

    # Display the graphs

    plt.figure(figsize = (30 , 10))
    plt.title("Predictions")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth
                = 2.0)
        plt.plot(np.arange(input.size(1), input.size(1)+future),
                 yi[input.size(1):], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.savefig(f"outputs/predict_{epoch}.png")
    plt.close()
