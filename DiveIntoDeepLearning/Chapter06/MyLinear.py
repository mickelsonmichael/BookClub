import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_units, units))

        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        # manually implement the linear regression step
        linear = torch.matmul(X, self.weight.data) + self.bias.data

        # ReLU activatio is baked-in to the module
        return nn.functional.relu(linear)
