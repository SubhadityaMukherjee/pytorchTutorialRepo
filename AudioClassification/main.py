import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from trainer import *
from tester import *
import torchvision.models as models
import os
os.environ["TORCH_HOME"] = "~/Desktop/Datasets/"

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet18',
                    help='''Choose any model from pytorch. Or input "my" for taking a model from model.py ''')
parser.add_argument("--weight-decay", default=1e-4,
                    help="weight decay coefficient")
parser.add_argument("--resume", default=False,
                    help="Resume training from a checkpoint")
parser.add_argument("--pretrained", default=True,
                    help="If part of the standard datasets, downloaded pretrained weights")
parser.add_argument('--batch-size', type=int,
                    default=128, help='input batch size')
parser.add_argument(
    "--test-batch-size", type=int, default=1000
)

parser.add_argument(
    "--epochs", type=int, default=20, help="no of epochs to train for"
)

parser.add_argument(
    "--lr", type=float, default=0.01, help="Base learning rate"
)

parser.add_argument(
    "--max_lr", type=float, default=0.1, help="Max learning rate for OneCycleLR"
)


parser.add_argument(
    "--dry-run", action='store_true', default=False, help='quickly check a single pass'
)

parser.add_argument(
    "--seed", type=int, default=100, help="torch random seed"
)

parser.add_argument(
    "--log-interval", type=int, default=20, help="interval to show results"
)

parser.add_argument(
    "--save-model", action='store_true', default=True, help="Choose if model to be saved or not"
)

parser.add_argument("--save_path", default="models/model.pt",
                    help="Choose model saved filepath")

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'batch_size': args.batch_size}
kwargs.update(
    {'num_workers': 8,
     'pin_memory': True,
     'shuffle': True

     }
)

# Getting the metadta

df = pd.read_csv(
    "/home/eragon/Documents/datasets/UrbanSound8K/metadata/UrbanSound8K.csv")
classes = list(df["class"].unique())

# Generate Mel Frequency Cepstrum


def extract_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(audio, sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


# Making custom dataset
# features = []
# labels = []
# folds = []
# print("[INFO] Making dataset")
# for i in range(len(df)):
#     fold = df["fold"].iloc[i]
#     filename = df["slice_file_name"].iloc[i]
#     path = "/home/eragon/Documents/datasets/UrbanSound8K/audio/fold{0}/{1}".format(
#         fold, filename)
#     mfccs = extract_mfcc(path)
#
#     # dataset.append((mfccs,df["classID"].iloc[i]))
#     features.append(mfccs)
#     folds.append(fold)
#     labels.append(df["classID"].iloc[i])
#
# features = torch.tensor(features)
# labels = torch.tensor(labels)
# folds = torch.tensor(folds)
#
# print("[INFO] Done making dataset")
# # Save to disk
# torch.save(
#     features, "/home/eragon/Documents/datasets/UrbanSound8K/features_mfccs.pt")
# torch.save(labels, "/home/eragon/Documents/datasets/UrbanSound8K/labels.pt")
# torch.save(folds, "/home/eragon/Documents/datasets/UrbanSound8K/folds.pt")
#
features = torch.load("/home/eragon/Documents/datasets/UrbanSound8K/features_mfccs.pt")
labels = torch.load("/home/eragon/Documents/datasets/UrbanSound8K/labels.pt")
folds = torch.load("/home/eragon/Documents/datasets/UrbanSound8K/folds.pt")


# According to the 8k Dataset website...


def get_dataset(skip_fold):
    local_features = []
    local_labels = []
    for i in range(len(folds)):
        if folds[i] == skip_fold:
            continue
        local_features.append(features[i])
        local_labels.append(labels[i])
    local_features = torch.stack(local_features)
    local_labels = torch.stack(local_labels)
    return TensorDataset(local_features, local_labels)

# Loading dataset


dataset = get_dataset(skip_fold=10)
print("Length of dataset: ", len(dataset))

val_size = int(0.1*len(dataset))
train_size = len(dataset) - val_size

train_data, test_data = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

# Loading model

if args.arch == "my":
    from Nets import *
    model = Net().to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

model = model.to(device)
print(model)
start_epoch = 1
if args.resume:
    loc = "cuda:0"
    checkpoint = torch.load(args.save_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])
    mode.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader), epochs=10)

for epoch in tqdm(range(start_epoch, args.epochs+1)):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), args.save_path)
