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
sys.path.append("/media/hdd/github/sprintdl/")
# -

from sprintdl.main import *
from sprintdl.nets import *
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda',0)
import torch
import math

# # Define required

# +
fpath = Path("/media/hdd/Datasets/asl/asl_alphabet_train/")

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 64
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-2], proc_y=CategoryProcessor())

ll

n_classes = len(set(ll.train.y.items)); n_classes

data = ll.to_databunch(bs, c_in=3, c_out=2)

show_batch(data, 4)

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
# arch = partial(xresnet18, n_classes)
arch = get_vision_model("resnet34", n_classes=n_classes, pretrained=True)

# opt_func = partial(sgd_mom_opt, wd=0.01)
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# opt_func = lamb
# -

# # Training

clear_memory()

# learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)
learn = Learner(arch,  data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

# +
# model_summary(learn, data)
# -

learn.fit(1)

save_model(learn, "m1", fpath)

# +
temp = Path('/media/hdd/Datasets/ArtClass/Popular/artgerm/10004370_1657536534486515_1883801324_n.jpg')

get_class_pred(temp, learn ,ll, 128)
# -

temp = Path('/home/eragon/Downloads/Telegram Desktop/IMG_1800.PNG')

get_class_pred(temp, learn ,ll,128)

temp = Path('/home/eragon/Downloads/Telegram Desktop/IMG_20210106_180731.jpg')

get_class_pred(temp, learn ,ll,128)

# # Digging in

classification_report(learn, n_classes, device)

learn.recorder.plot_lr()

learn.recorder.plot_loss()





# # Xresnet

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
arch = partial(xresnet18, c_out = n_classes)
# arch = get_vision_model("resnet34", n_classes=n_classes, pretrained=True)

# opt_func = partial(sgd_mom_opt, wd=0.01)
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# opt_func = lamb
# -

# # Training

clear_memory()

# learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)
learn = Learner(arch(),  data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

# +
# model_summary(learn, data)
# -

learn.fit(1)

save_model(learn, "m2", fpath)

# # Digging in

classification_report(learn, n_classes, device)

learn.recorder.plot_lr()

learn.recorder.plot_loss()


