import torch
from d2l import torch as d2l

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()

        self.save_hyperparameters()

        self.W = torch.normal(
            0,
            sigma,
            size=(num_inputs, num_outputs),
            requires_grad=True
        )

        self.b = torch.zeros(num_outputs, requires_grad=True)