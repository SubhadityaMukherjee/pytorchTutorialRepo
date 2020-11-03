import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper

# 
# @torchsnooper.snoop()
class Generator(torch.nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(num_classes, num_classes)
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(100+num_classes, 128, normalize=False), # 100 -> latent dim ; This is new
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # This is new; Concat embedding of label + image -> input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes) # This is new
        
        self.model = nn.Sequential( # This is new  
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 1),
        )

    def forward(self, x, labels):
        img_flat = torch.cat((x.view(x.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(img_flat)
        return validity
import hiddenlayer as hl
from torch.autograd import Variable
x = Variable(torch.rand(1, 1, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save('gp.png')
