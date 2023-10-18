#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#configuration
DEVICE = 'cpu'

# LSTM model definition
class NetworkLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_stacked_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_stacked_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, 1)

        self.sig = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_dim).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.sig(out)
        return out

# Dataset definition
class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]