import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype = torch.double)

        c_t = torch.zeros(input.size(0), 51, dtype = torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype = torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype = torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim = 1)):
            h_t , c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t , (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in tqdm(range(future)):
            h_t, c_t = self.lstm1(output, (h_t , c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
import hiddenlayer as hl
from torch.autograd import Variable
x = Variable(torch.rand(1, 1, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save('gp.png')
