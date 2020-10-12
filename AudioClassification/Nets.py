import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.ReLU(),

            nn.Linear(64, 10),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)
