# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
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

from sprintdl.main import *
import sprintdl

device = torch.device("cuda", 0)
from torch.nn import init
import torch
import math

from sprintdl.models.efficientnet import *
# -

# # Define required

import pandas as pd

fpath = Path(
    "/media/hdd/Datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv"
)

df = pd.read_csv(fpath)
df.head(100)

df.shape

df = df[df["Dataset_type"] == "TRAIN"]

df.to_csv(
    "/media/hdd/Datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train.csv"
)

len(df.Label.unique())

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256

# +
lr = 1e-2
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
    partial(CudaCallback, device),
]

loss_func = LabelSmoothingCrossEntropy()
lr = 0.001
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# -

tl = TableLoader(
    "/media/hdd/Datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train.csv",
    "X_ray_image_name",
    "Label",
    add_before="/media/hdd/Datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/",
)

il = ImageList(tl, tfms=tfms)

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.2))

ll = label_by_func(sd, partial(table_labeler, dic=tl), proc_y=CategoryProcessor())

data = ll.to_databunch(bs, c_in=3, c_out=2)

show_batch(data)

# # Xresnet34 + pruning

arch = partial(xresnet34, c_out=2)()

learn = Learner(
    arch, data, loss_func, lr=lr, cb_funcs=cbfs + [PruningCallback], opt_func=opt_func
)

learn.fit(1)

learn.fit(2)

learn.fit(2)

# # Xresnet34

arch = partial(xresnet34, c_out=2)()

learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)

learn.fit(2)

learn.fit(2)



# # Effnet

from sprintdl.models.efficientnet import *

arch = efficientnet(2, pretrained=False)

clear_memory()

# +
# learn.destroy()
# -

data = ll.to_databunch(64, c_in=3, c_out=2)

learn = Learner(
    arch, data, loss_func, lr=lr, cb_funcs=cbfs + [PruningCallback], opt_func=opt_func
)

learn.fit(1)

learn.fit(4)

learn.fit(2)

learn.fit(2)



# # Effnet + Pruning

from sprintdl.models.efficientnet import *

arch = efficientnet(2, pretrained=False)

clear_memory()

# +
# learn.destroy()
# -

data = ll.to_databunch(64, c_in=3, c_out=2)

learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)

learn.fit(4)

learn.fit(2)

learn.fit(2)

learn.fit(2)
