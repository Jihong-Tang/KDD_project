import numpy as np

import torch
from torch import nn

from torchsummary import summary


# input: [batch_size, 96, 26]
# output: [2]
class LSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: int,
                 num_classes: int, device: torch.device):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    model = LSTM(26, 64, 4, 0, 2, torch.device('cpu')).float()
    summary(model, (96, 26))
    # x = np.zeros([4, 96, 26])
    # x = torch.zeros([4, 100, 26])
    # out = model(x)
    # print(out)
    # print(out.shape)
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # out = torch.Tensor([
    #     [1, 0],
    #     [1, 0],
    #     [1, 0],
    #     [1, 0],
    # ])
    # print(criterion(out, torch.Tensor([0, 0, 0, 0]).long()))
    # print(criterion(out, torch.Tensor([1, 1, 1, 1]).long()))
    # _, predicted = torch.max(out.data, 1)
    # print(predicted)
