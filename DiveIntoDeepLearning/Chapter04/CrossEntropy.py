import torch

# Cross-Entropy Loss function
# <https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html#the-cross-entropy-loss>
def CrossEntropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()
