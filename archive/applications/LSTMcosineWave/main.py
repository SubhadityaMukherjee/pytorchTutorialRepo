import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from trainer import *
from tester import *
import torchvision.models as models
import os

os.environ["TORCH_HOME"] = "~/Desktop/Datasets/"

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="folder for custom training", default="")
parser.add_argument(
    "--arch",
    default="resnet18",
    help="""Choose any model
                    from pytorch. Or input "my" for taking a model from
                    model.py """,
)
parser.add_argument("--weight-decay", default=1e-4, help="weight decay coefficient")
parser.add_argument("--resume", default=False, help="Resume training from a checkpoint")
parser.add_argument(
    "--pretrained",
    default=True,
    help="If part of the standard datasets, downloaded pretrained weights",
)
parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
parser.add_argument("--test-batch-size", type=int, default=1000)

parser.add_argument("--epochs", type=int, default=20, help="no of epochs to train for")

parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate")

parser.add_argument(
    "--max_lr", type=float, default=0.1, help="Max learning rate for OneCycleLR"
)


parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)

parser.add_argument("--seed", type=int, default=100, help="torch random seed")

parser.add_argument(
    "--log-interval", type=int, default=20, help="interval to show results"
)

parser.add_argument(
    "--save-model",
    action="store_true",
    default=True,
    help="Choose if model to be saved or not",
)

parser.add_argument(
    "--save_path", default="models/model.pt", help="Choose model saved filepath"
)

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {"batch_size": args.batch_size}
kwargs.update({"num_workers": 8, "pin_memory": True, "shuffle": True})


# Loading dataset

data = torch.load("models/traindata.pt")
input = torch.from_numpy(data[3:, :-1])
target = torch.from_numpy(data[3:, 1:])
test_input = torch.from_numpy(data[:3, :-1])
test_target = torch.from_numpy(data[:3, 1:])

# Loading model

if args.arch == "my":
    from Nets import *

    model = Net()
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()
model.double()
print(model)
start_epoch = 1
if args.resume:
    loc = "cuda:0"
    checkpoint = torch.load(args.save_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    mode.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

optimizer = optim.LBFGS(model.parameters(), lr=0.8)

for epoch in tqdm(range(start_epoch, args.epochs + 1)):
    train(
        args,
        model,
        device,
        data,
        input,
        target,
        test_input,
        test_target,
        optimizer,
        epoch,
    )

if args.save_model:
    torch.save(model.state_dict(), args.save_path)
