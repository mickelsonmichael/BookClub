import torch
from torch import nn # nn = neural-network
from d2l import torch as d2l
from relu import relu

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()

        self.save_hyperparameters()

        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.bw = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        # reshape the n x n images into flat vectors
        X = X.reshape((-1, self.num_inputs))

        # get the value of the hidden layer
        H = relu(torch.matmul(X, self.W1) + self.b1)

        # get the output value
        return torch.matmul(H, self.W2) + self.b2
    