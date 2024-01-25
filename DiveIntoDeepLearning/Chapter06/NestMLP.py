from torch import nn


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU()
        )

        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))
