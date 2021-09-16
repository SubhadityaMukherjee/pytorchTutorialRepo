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

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
import sys

sys.path.append("/media/hdd/github/sprintdl/")
# -

from sprintdl.main import *
import sprintdl

device = torch.device("cuda", 0)
from torch.nn import init
import torch
import math

from sprintdl.models.efficientnet import *

# # Define required

# +
fpath = Path("/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/")

# train_transform = [A.Resize(128,128)]

# tfms = [ATransform(train_transform, c_in = 3)]
tfms = [make_rgb, to_byte_tensor, to_float_tensor, ResizeFixed(128)]
bs = 100
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

il

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-3], proc_y=CategoryProcessor())

n_classes = len(set(ll.train.y.items))
n_classes

data = ll.to_databunch(bs, c_in=3, c_out=n_classes)

show_batch(data, 4)

# # Training

# +
lr = 1e-3
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr = combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgStatsCallback, accuracy),
    partial(ParamScheduler, "lr", sched_lr),
    partial(ParamScheduler, "mom", sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
    OverSampling,
    partial(CudaCallback, device),
]

loss_func = LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# -

arch = efficientnet(num_classes=n_classes, pretrained=True)

# # OTher loss

loss_func = nll

clear_memory()
learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)
learn.fit(2)

learn.fit(4)

# # Label smooth CE

clear_memory()
learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)
learn.fit(2)

learn.fit(2)
