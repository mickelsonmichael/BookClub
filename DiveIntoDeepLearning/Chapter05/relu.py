import torch

def relu(X):
    # https://pytorch.org/docs/stable/generated/torch.zeros_like.html
    # Create a tensor with the same dimension as X, but with zeros
    a = torch.zeros_like(X)

    return torch.max(X, a)