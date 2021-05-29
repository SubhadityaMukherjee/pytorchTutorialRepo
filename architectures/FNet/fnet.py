# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

import os
os.environ['TORCH_HOME'] = "/media/hdd/Datasets/"
import sys
sys.path.append("../../../sprintdl/")
# -

from sprintdl.main import *
import sprintdl

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

# # Define required

# +
fpath = Path("/media/hdd/Datasets/imagewoof2-160/")

train_transform = [A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]

tfms = [ATransform(train_transform, c_in = 3)]
bs = 256
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

il

tm= Path("/media/hdd/Datasets/ArtClass/Unpopular/mimang.art/69030963_140928767119437_3621699865915593113_n.jpg")

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-3], proc_y=CategoryProcessor())

n_classes = len(set(ll.train.y.items))

data = ll.to_databunch(bs, c_in=3, c_out=2)

show_batch(data, 4)

# # Training

# +
lr = .001
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
#     MixUp,
       partial(CudaCallback, device)]

loss_func=LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# -

# # FNet
# - https://github.com/rishikksh20/FNet-pytorch/blob/master/fnet.py

# +
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout),
#         )
#     def forward(self,x):
#         return self.net(x)

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self,x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FNetBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,x):
#         x = torch.fft.fft(torch.fft.fft(x, dim -1), dim = -2).real
#         return x

# class FNet(nn.Module):
#     def __init__(self,dim, depth, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#             nn.ModuleList([
#                 PreNorm(dim, FNetBlock()),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

# -

from sprintdl.models.fnet import *

# arch = partial(xresnet34, c_out =n_classes)()
arch = FNet(3, 3, 10)

arch

count_parameters(arch)

learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)




