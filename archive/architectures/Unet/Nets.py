import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from torchsummary import summary

# Conv -> Relu

# This is new
def conv_layer(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


# Main network


class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])

        self.layer0_1x1 = conv_layer(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])

        self.layer1_1x1 = conv_layer(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = conv_layer(128, 128, 1, 0)

        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = conv_layer(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = conv_layer(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up3 = conv_layer(256 + 512, 512, 3, 1)
        self.conv_up2 = conv_layer(128 + 512, 256, 3, 1)
        self.conv_up1 = conv_layer(64 + 256, 256, 3, 1)
        self.conv_up0 = conv_layer(64 + 256, 128, 3, 1)

        self.conv_original_size0 = conv_layer(3, 64, 3, 1)
        self.conv_original_size1 = conv_layer(64, 64, 3, 1)
        self.conv_original_size2 = conv_layer(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        print(out.shape)
        out = out.permute(0, 3, 2, 1)

        return out


# Helper to generate model architecture
# If you are using linux/mac just run this to save it to a text file
"""
python -c "from Nets import *; arch_print >> architecture.txt"
"""


def arch_print(n_classes=3):
    ne = Net(n_classes).to("cuda")
    summary(ne.to("cuda"), input_size=(3, 224, 224))


import hiddenlayer as hl
from torch.autograd import Variable

x = Variable(torch.rand(1, 1, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save("gp.png")
