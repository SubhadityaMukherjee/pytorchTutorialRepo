import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def test(model, device, test_loader):
    model.eval()  # Setting model to test
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.cross_entropy(output, target)
            acc = accuracy(output, target)
            print(f"Acc: {acc}")
            # print(f"Val loss: {test_loss.detach()}, Val acc : {acc}")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
