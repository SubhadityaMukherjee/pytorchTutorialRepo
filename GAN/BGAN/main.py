import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from trainer import *
import torchvision.models as models
import os
os.environ["TORCH_HOME"] = "~/Desktop/Datasets/"

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--data', help = "folder for custom training", default = "")
parser.add_argument('--arch', default = 'resnet18', help= '''Choose any model
                    from pytorch. Or input "my" for taking a model from
                    model.py ''')
parser.add_argument("--weight-decay", default = 1e-4, help = "weight decay coefficient")
parser.add_argument("--resume", default = False, help = "Resume training from a checkpoint")
parser.add_argument("--pretrained", default = False, help = "If part of the standard datasets, downloaded pretrained weights")
parser.add_argument('--batch-size', type = int, default = 128, help = 'input batch size')
parser.add_argument(
    "--test-batch-size", type = int, default = 1000
)

parser.add_argument(
    "--epochs", type = int, default = 20, help = "no of epochs to train for"
)

parser.add_argument(
    "--lr", type = float, default = 0.0002, help = "Base learning rate"
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
    "--log_interval", type = int, default = 20, help = "interval to show results"
)

parser.add_argument(
    "--save-model", action = 'store_true', default = True, help = "Choose if model to be saved or not"
)

parser.add_argument("--save_path", default = "models/model.pt", help = "Choose model saved filepath")

# GAN specific args
 
parser.add_argument("--nz", default = 100, help = "size of latent vector")
parser.add_argument("--ngf", default = 28, help = "gen size")
parser.add_argument("--ndf", default = 28, help= "Discriminator size")
parser.add_argument("--beta1", default = 0.5, help = "adam beta1 parameter")
parser.add_argument('--CRITIC_ITERS', default=1, type=int, help='D update iters before G update')
parser.add_argument("--nc", default =1, help = "number of image channels")
args = parser.parse_args()

# Setting params

nz = int(args.nz)
nsamplesgf = int(args.ngf)
ndf = int(args.ndf)
nc = int(args.nc)

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
    [transforms.Resize(28),
    transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# Loading dataset

train_data = datasets.MNIST("~/Desktop/Datasets/", transform =
                            transform)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)

# Loading model

if args.arch == "my":
    from Nets import *
    netG = Generator((nc, ndf, ndf)).to(device)
    netD = Discriminator((nc, ndf, ndf)).to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

print("Generator", netG)
print("Discriminator", netD)

start_epoch = 1
if args.resume:
    loc = "cuda:0"
    checkpointD = torch.load(args.save_path+"dis.pt", map_location = loc)
    
    checkpointG = torch.load(args.save_path+"gen.pt", map_location = loc)
    netD.load_state_dict(checkpoint['state_dict'])
    netD.load_state_dict(checkpoint['optimizer'])
    netG.load_state_dict(checkpoint['state_dict'])
    netG.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch']

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

# Generate noise
# Loss weight for gradient penalty

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=args.lr,betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr,betas=(args.beta1, 0.999))

# Loop
batches_done = 0
for epoch in tqdm(range(start_epoch, args.epochs+1)):
    train(args, device, train_loader, epoch, netD, netG,nz , ndf, nc, optimizerD, optimizerG, batches_done) 


if args.save_model:

    torch.save(netD.state_dict(), args.save_path+"disc.pt")
    
    torch.save(netG.state_dict(), args.save_path+"gen.pt")





