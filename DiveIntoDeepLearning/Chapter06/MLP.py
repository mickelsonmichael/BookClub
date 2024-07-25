from torch import nn
from torch.nn import functional


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.LazyLinear(256)  # define the hidden layer
        self.out = nn.LazyLinear(10)  # define the output layer

    # forward is the "forward propogation" function
    # returns the result of the model (output layer) in regards to X
    def forward(self, X):
        return self.out(functional.relu(self.hidden(X)))
