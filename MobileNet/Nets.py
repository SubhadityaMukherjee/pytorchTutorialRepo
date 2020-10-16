import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.lin1 = nn.Linear(1024, 100)
        self.convn = nn.Sequential(  # depthwise when changing from a smaller to bigger layer
            self.conv_block(32, 64, 1),
            self.conv_block(64, 128, 2),
            self.conv_block(128, 128, 1),
            self.conv_block(128, 256, 2),
            self.conv_block(256, 256, 1),
            self.conv_block(256, 512, 2),
            self.conv_block(512, 512, 1),
            self.conv_block(512, 512, 1),
            self.conv_block(512, 512, 1),
            self.conv_block(512, 512, 1),
            self.conv_block(512, 512, 1),
            self.conv_block(512, 1024, 2),
            self.conv_block(1024, 1024, 1),
            nn.AvgPool2d(4),
        )

    def conv_block(self, inb, out, stride):
        return nn.Sequential(
            nn.Conv2d(
                inb, inb, 3, stride, 1, groups=inb, bias=False
            ),  # groups -> no of blocks -> depthwise conv
            nn.BatchNorm2d(inb),
            nn.ReLU(inplace=True),
            nn.Conv2d(inb, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.convn(x)
        x = x.view(-1, 1024)
        x = self.lin1(x)

        return x
