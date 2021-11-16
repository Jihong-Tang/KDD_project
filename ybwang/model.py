import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchviz import make_dot


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        # inputs (batch_size,96,31)
        self.encoder0 = nn.GRU(input_size=96, hidden_size=96, num_layers=2, dropout=0.1, batch_first=True)
        self.fc0 = nn.Linear(96, 1)

        self.encoder = nn.GRU(input_size=31, hidden_size=31, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(31, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        out0, hn0 = self.encoder0(tensor)
        result0 = self.fc0(out0)
        result0 = result0.permute(0, 2, 1)
        out, hn = self.encoder(result0)
        result = self.fc(out)
        return result

if __name__ == '__main__':
    x=torch.rand(10,96,31)
    gru=GRUModel()
    y=gru(x)
    g=make_dot(y)
    g.render('GRU_model',view=True)
