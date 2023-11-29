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
        self.args = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'num_stacked_layers': num_stacked_layers}

        self.hidden_dim = hidden_dim
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_stacked_layers, batch_first=True, proj_size=0)

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_dim).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = torch.abs(out)
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
    

"""
Transform list of conversations to pytorch tensors
"""
def list_tensor(x, conv_len, input_dim):
    if conv_len > 0:
        ret = torch.tensor(())
        tmp = torch.tensor((), dtype=torch.float32)

        for conv in range(len(x)):
            tmp = tmp.new_zeros(1, conv_len, input_dim)

            for msg in range(len(x[conv])):
                for input in range(input_dim):
                    tmp[0][msg][input] = float(x[conv][msg][input])

            ret = torch.cat((ret, tmp))
        return ret
    else:
        ret = torch.tensor(())
        ret = ret.new_zeros(1, 1, input_dim)
        return ret


"""
NN checkpoint creation
"""
def checkpoint(model, filename):
    torch.save([model.args, model.state_dict()], filename)


"""
NN checkpoint load
"""
def resume(model, filename):
    _, state = torch.load(filename)
    model.load_state_dict(state)