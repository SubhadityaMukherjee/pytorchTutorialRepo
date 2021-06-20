# Generate cosine wave

import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), "int64")
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.cos(x / 1.0 / T).astype("float64")
print(data)
torch.save(data, open("models/traindata.pt", "wb+"))
