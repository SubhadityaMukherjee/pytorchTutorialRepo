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
from sklearn.neighbors import KNeighborsClassifier
import os

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"


# -

# # Train

class LinearRegressionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearRegressionModel, self).__init__()

        # init the layers
        self.layers = nn.Sequential(
            # input linear layer
            nn.Linear(1, 10),
            # "rectified linear unit" or "replace negatives with zeroes"
            nn.ReLU(inplace=True),
            # batch norm
            nn.BatchNorm1d(num_features=10),
            # output linear layer
            nn.Linear(10, 1),
        )

    # fastai passes both x_cat and x_cont - we can just ignore x_cat
    def forward(self, x_cont: Tensor, *args, **kwargs):
        # pass x_cont into the network
        return self.layers(x_cont);


root_dir = "/media/hdd/Datasets/boat/"
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

learn = vision_learner(
    dls, LinearRegressionModel(), metrics=[accuracy, error_rate], pretrained=False
)
learn.fit_one_cycle(1)




