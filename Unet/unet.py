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

# # Modules

# +
import timm
from fastai.vision.all import *
from fastai.vision.widgets import *
import os

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"
# -

# # Train

root_dir = "/media/hdd/Datasets/DenseHaze/"
path = Path(root_dir)

path.ls()


def file_repl(x):
    if "masks" in str(x):
        return Path(str(x).replace("masks", "train"))
    else:
        return Path(str(x).replace("train", "masks"))


# +
def ret(o):
    return o.parent.parent / "masks" / o.name


ret(Path("/media/hdd/Datasets/DenseHaze/train/43.png"))
# -

fields = DataBlock(
    blocks=(ImageBlock(cls=PILImage), ImageBlock(cls=PILImage)),
    get_items=get_image_files,
    #     get_y= lambda o:o.parent.parent/'train'/o.name,
    get_y=lambda o: o,
    #     splitter=GrandparentSplitter(),
    splitter=RandomSplitter(),
    item_tfms=Resize(32),
    #     batch_tfms=aug_transforms(),
)
dls = fields.dataloaders(path, path=path, bs=10)

dls.show_batch(max_n=3)

# +
arch = create_body(xresnet18, n_in=1).cuda()


class UpsampleBlock(Module):
    def __init__(
        self,
        up_in_c: int,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        **kwargs
    ):
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        ni = up_in_c // 2
        nf = ni if final_div else ni // 2
        self.conv1 = ConvLayer(ni, nf, **kwargs)
        self.conv2 = ConvLayer(nf, nf, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, up_in: Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        cat_x = self.relu(up_out)
        return self.conv2(self.conv1(cat_x))


def decoder_resnet(y_range, n_out=1):
    return nn.Sequential(
        UpsampleBlock(512),
        UpsampleBlock(256),
        UpsampleBlock(128),
        UpsampleBlock(64),
        UpsampleBlock(32),
        nn.Conv2d(16, n_out, 1),
        SigmoidRange(*y_range),
    )


def autoencoder(encoder, y_range):
    return nn.Sequential(encoder, decoder_resnet(y_range))


y_range = (-3.0, 3.0)
ac_resnet = autoencoder(arch, y_range).cuda()
dec = decoder_resnet(y_range).cuda()
# -

learn = Learner(dls, ac_resnet, loss_func=MSELossFlat())
learn.fit_one_cycle(3)




