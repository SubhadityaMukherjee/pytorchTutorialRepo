# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] heading_collapsed=true
# # Modules

# + hidden=true
import timm
from fastai.vision.all import *
from fastai.vision.widgets import *
import os

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"

# + [markdown] heading_collapsed=true
# # Train

# + hidden=true
root_dir = "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/"
path = Path(root_dir)
fields = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(),
)
dls = fields.dataloaders(path)
learn = vision_learner(dls, resnet34, metrics=[accuracy, error_rate]).to_fp16()
learn.fine_tune(1)


# -

# # GradCAM

class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class HookBwd:
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)

    def hook_func(self, m, gi, go):
        self.stored = go[0].detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


import matplotlib.pyplot as plt
from IPython.display import Image

# +
img = PILImage.create(
    "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/Shrimp/Shrimp/00012.png"
)
(x,) = first(dls.test_dl([img]))
# cam_map = torch.einsum('ck,kij->cij', learn.model[1][-1].weight, act)
x_dec = TensorImage(dls.train.decode((x,))[0][0])

image_count = len(learn.model[0])
col = 4
row = math.ceil(image_count / col)
plt.figure(figsize=(col * 4, row * 4))
plt.figure(figsize=(col * 4, row * 4))

for layer in range(image_count):  # no of layers
    cls = 1
    try:
        with HookBwd(learn.model[0][layer]) as hookg:  # for other layers
            with Hook(learn.model[0][layer]) as hook:
                output = learn.model.eval()(x.cuda())
                act = hook.stored
            output[0, cls].backward()
            grad = hookg.stored
        w = grad[0].mean(dim=[1, 2], keepdim=True)
        cam_map = (w * act[0]).sum(0)

    except:
        pass

    plt.subplot(row, col, layer + 1)
    x_dec.show(ctx=plt)
    plt.imshow(
        cam_map.detach().cpu(),
        alpha=0.8,
        extent=(0, 224, 224, 0),
        interpolation="bilinear",
        cmap="magma",
    )
    plt.title(f"Layer : {layer}")
    plt.axis("off")
# -




