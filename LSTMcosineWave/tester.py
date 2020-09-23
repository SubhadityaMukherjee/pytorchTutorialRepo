import torch
import torch.nn as nn

criterion = nn.MSELoss()

def test(model, test_input, test_target):
    with torch.no_grad():
        future = 1000
        pred = model(test_input, future = future)
        loss = criterion(pred[:, :-future], test_target)
        print(f"Test loss: {loss.item()}")
        y = pred.detach().numpy()
    return y,future
