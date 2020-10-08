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
from data import *

os.environ["TORCH_HOME"] = "~/Documents/datasets/"

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--data', help="folder for custom training", default="")
parser.add_argument('--arch', default='resnet18', help='''Choose any model
                    from pytorch. Or input "my" for taking a model from
                    model.py ''')
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

parser.add_argument(
    "--data_path", default="/home/eragon/Documents/datasets/bw2color", help="dataset path")

parser.add_argument("--beta1", default=0.5, help="Adam beta1 parameter")

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


# Loading dataset

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = Data(350, args.data_path, trans)
test_set = Data(50, args.data_path, trans)

train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

# Loading model

if args.arch == "my":
    from Nets import *
    model = Net(3).to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

print("Model ", model)

start_epoch = 1
if args.resume:
    loc = "cuda:0"
    model = torch.load(args.save_path, map_location=loc)
    print("Done loading model")

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       betas=(args.beta1, 0.999))

# Loop
#criterion = nn.MSELoss()


def diceloss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    #print(pred.shape, target.shape)
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


for epoch in tqdm(range(start_epoch, args.epochs+1)):
    train(args, device, train_loader, model, epoch, optimizer, diceloss)

if args.save_model:
    torch.save(model, args.save_path)
