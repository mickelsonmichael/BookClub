import torch
from torch import nn


def comp_conv2d(conv2d, X):
    """
    Calculates convolutions

    :param conv2d: A function to perform a convolutional calculation
    :param X: The input data
    """

    # Reshapes the 8x8 "image" into a 1x1x8x8
    # where batch size is 1
    # and number of channels is 1
    X = X.reshape((1, 1) + X.shape)

    Y = conv2d(X)

    # Remove the batch and channel sizes
    return Y.reshape(Y.shape[2:])