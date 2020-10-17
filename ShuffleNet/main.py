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
parser.add_argument('--arch', default = 'resnet18', help= '''Choose any model from pytorch. Or input "my" for taking a model from model.py ''')
parser.add_argument("--weight-decay", default = 1e-4, help = "weight decay coefficient")
parser.add_argument("--resume", default = False, help = "Resume training from a checkpoint")
parser.add_argument("--pretrained", default = True, help = "If part of the standard datasets, downloaded pretrained weights")
parser.add_argument('--batch-size', type = int, default = 128, help = 'input batch size')
parser.add_argument(
    "--test-batch-size", type = int, default = 1000
)

parser.add_argument(
    "--epochs", type = int, default = 20, help = "no of epochs to train for"
)

parser.add_argument(
    "--lr", type = float, default = 0.01, help = "Base learning rate"
)

parser.add_argument(
    "--max_lr", type = float, default = 0.1, help = "Max learning rate for OneCycleLR"
)


parser.add_argument(
    "--dry-run", action = 'store_true', default = False, help = 'quickly check a single pass'
)

parser.add_argument(
    "--seed", type = int, default = 100, help = "torch random seed"
)

parser.add_argument(
    "--log-interval", type = int, default = 20, help = "interval to show results"
)

parser.add_argument(
    "--save-model", action = 'store_true', default = True, help = "Choose if model to be saved or not"
)

parser.add_argument("--save_path", default = "models/model.pt", help = "Choose model saved filepath")

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'batch_size':args.batch_size}
kwargs.update(
    {'num_workers':8,
     'pin_memory':True,
     'shuffle': True

    }
)

# Defining batch transforms

transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
    ]
)

# Loading dataset

train_data = datasets.CIFAR10("~/Desktop/Datasets/",train = True, transform = transform)

test_data = datasets.CIFAR10("~/Desktop/Datasets/",train = False, transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

# Loading model

if args.arch == "my":
    from Nets import *
    model = Net(n_class=10).to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

model = model.to(device)
print(model)
start_epoch = 1
if args.resume:
    loc = "cuda:0"
    checkpoint = torch.load(args.save_path, map_location = loc)
    model.load_state_dict(checkpoint['state_dict'])
    mode.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay =
                        args.weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr =
                                       args.max_lr,steps_per_epoch =
                                          len(train_loader), epochs = 10)

for epoch in tqdm(range(start_epoch, args.epochs+1)):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), args.save_path)





