import torch
from torch import nn
from corr2d import corr2d

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init()

        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
