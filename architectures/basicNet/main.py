import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model import *
from trainer import *
from tester import *

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
parser.add_argument("--test-batch-size", type=int, default=1000)

parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--lr", type=float, default=0.1)

parser.add_argument("--gamma", type=float, default=0.7)

parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)

parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--log-interval", type=int, default=10)

parser.add_argument("--save-model", action="store_true", default=True)

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {"batch_size": args.batch_size}
kwargs.update({"num_workers": 8, "pin_memory": True, "shuffle": True})

# Defining batch transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Loading dataset

train_data = datasets.CIFAR10("~/Desktop/Datasets/", train=True, transform=transform)

test_data = datasets.CIFAR10("~/Desktop/Datasets/", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

# Loading model

model = Net().to(device)
print(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=0.1, gamma=args.gamma)

for epoch in tqdm(range(1, args.epochs + 1)):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

checkpoint = {
    "epoch": epoch + 1,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

if args.save_model:
    torch.save(checkpoint, "./models/model.pt")
