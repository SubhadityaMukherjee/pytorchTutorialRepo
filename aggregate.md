# FederatedLearningPySyft/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# AudioClassification/Nets.py
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


# SeLU/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# This is new
class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()
        self.α = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.α * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2


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
            SELU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            SELU(),
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
                SELU(),
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(out, out, kernel_size=3, padding=1, stride=1),
                    SELU(),
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


# ShuffleNet/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

# This is new
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

# This is new
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, : (x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2) :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class Net(nn.Module):
    def __init__(self,n_class=10, input_size=224, width_mult=1.0):
        super(Net, self).__init__()

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(
                    num_groups
                )
            )

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, 2, 2)
                    )
                else:
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, 1, 1)
                    )
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(1))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


# AdversarialFGSM/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

#Le net 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# MobileNet/Nets.py
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


# Pruning/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Le net 5 (Le Cun et al 1998)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 channel inp, 6 outputs, 3x3 conv
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120) #5x5 imgs
        self.fc2 = nn.Linear(129, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement()/x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# BatchnormOnlyBatchnorm/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,1)
        self.conv2 = nn.Conv2d(32, 64, 3,1)
        self.dropout1 = nn.Dropout2d(.25)
        self.dropout2 = nn.Dropout2d(.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.bn3(F.relu(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output


# VAE/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    # This is new
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# FocalLoss/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

    def forward(self, x):
        pass


# SuperRes/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Main network
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5,5), (1,1), (2,2))
        self.conv2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.conv3 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3,3), (1,1), (1,1))
        # This is new
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor) 

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight,init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight,init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)



# SqueezeNet/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# This is new
class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(
            squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)  # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
#         x = torch.flatten(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
        return x        


# STN/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# This is new
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Mish/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# This is new
class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        return x*torch.sigmoid(x) 

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
            mish(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            mish(),
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
                mish(),
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(out, out, kernel_size=3, padding=1, stride=1),
                    mish(),
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


# VGG16/Nets.py
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


# AlexNet/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Unet/Nets.py
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

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = conv_layer(256+512, 512, 3, 1)
        self.conv_up2 = conv_layer(128+512, 256, 3, 1)
        self.conv_up1 = conv_layer(64+256, 256, 3, 1)
        self.conv_up0 = conv_layer(64+256, 128, 3, 1)

        self.conv_original_size0 = conv_layer(3, 64, 3, 1)
        self.conv_original_size1 = conv_layer(64, 64, 3, 1)
        self.conv_original_size2 = conv_layer(64+128, 64, 3, 1)

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
    ne = Net(n_classes).to('cuda')
    summary(ne.to('cuda'), input_size=(3, 224, 224))


# standardModels/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(.25)
        self.dropout2 = nn.Dropout2d(.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output


# Swish/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# This is new
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
    def forward(self, x):
        return x*torch.sigmoid(x)

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
            swish(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            swish(),
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
                swish(),
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(out, out, kernel_size=3, padding=1, stride=1),
                    swish(),
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


# LSTMcosineWave/Nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype = torch.double)

        c_t = torch.zeros(input.size(0), 51, dtype = torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype = torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype = torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim = 1)):
            h_t , c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t , (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in tqdm(range(future)):
            h_t, c_t = self.lstm1(output, (h_t , c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


