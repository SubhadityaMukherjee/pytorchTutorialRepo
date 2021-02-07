import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = self.conv_block(2, 3, 64)
        self.conv2 = self.conv_block(2, 64, 128)
        self.conv3 = self.conv_block(3, 128, 256)
        self.conv4 = self.conv_block(3, 256, 512)
        self.conv5 = self.conv_block(3, 512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classif = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(4096, 100),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def conv_block(self, n, inb, out):
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(inb, out, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(out, out, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n - 1)
            ],
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        self._initialize_weights()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classif(x)

        return x
import hiddenlayer as hl
from torch.autograd import Variable
x = Variable(torch.rand(1, 1, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save('gp.png')
