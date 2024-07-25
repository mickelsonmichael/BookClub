from torch import nn
from d2l import torch as d2l

# <https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html#defining-the-model>
class SoftmaxRegression(d2l.Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()

        self.save_hyperparameters()

        # Create a sequential Neural Net
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs)
        )

    def forward(self, X):
        return self.net(X)
