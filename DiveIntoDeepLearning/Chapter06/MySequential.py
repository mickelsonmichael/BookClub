from torch import nn


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)

        return X
