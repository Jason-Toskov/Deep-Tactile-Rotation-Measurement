
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
class RegressionLSTM(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout, seq_length=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # self.start_linear1 = nn.Linear(in_features=self.num_features, out_features=200)
        # self.start_linear2 = nn.Linear(in_features=200, out_features=200)
        # self.start_linear3 = nn.Linear(in_features=200, out_features=self.num_features)
        # self.sig = nn.Sigmoid()

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout
        )

        # self.out_lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.out_lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size)
        self.output_linear_final = nn.Linear(
            in_features=self.hidden_size, out_features=1)

    def init_model_state(self, batch_size):

        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        return (h0, c0)

    def forward(self, x, h0=None, c0=None):

        batch_size = x.shape[0]

        if h0 is None or c0 is None:
            h0, c0 = self.init_model_state(batch_size)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        out = self.output_linear(out)
        out = self.output_linear_final(out)


        return out, h_n, c_n