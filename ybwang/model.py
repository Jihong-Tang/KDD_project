import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        # inputs (batch_size,96,31)
        self.encoder = nn.GRU(input_size=31, hidden_size=31, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(31, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, tensor):
        out, hn = self.encoder(tensor)
        result = self.fc(out)
        return result
