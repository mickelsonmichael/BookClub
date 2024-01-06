import torch

# A rudimentary Softmax function which converts a list of scalars
# into a list of probabilities.
# <https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html#the-softmax>
def softmax(X):
    X_exp = torch.exp(X)

    partition = X_exp.sum(dim=1, keepdim=1)

    return X_exp / partition
