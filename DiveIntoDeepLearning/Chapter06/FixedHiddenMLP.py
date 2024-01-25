import torch
from torch import nn


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # random constant for each instance
        self.random_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)

        # @ operator is matrix multiplication
        X = nn.functional.relu(X @ self.random_weight + 1)

        # user the linear layer twice
        X = self.linear(X)

        while X.abs().sum() > 1:
            X /= 2

        return X.sum()
